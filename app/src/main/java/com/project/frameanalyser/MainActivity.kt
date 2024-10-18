package com.project.frameanalyser

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.core.content.ContextCompat
import com.google.mediapipe.components.CameraHelper
import com.google.mediapipe.components.CameraXPreviewHelper
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.project.frameanalyser.databinding.ActivityMainBinding
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.ProtocolException
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.Volatile


typealias LumaListener = (luma: Double) -> Unit

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    private lateinit var cameraExecutor: ExecutorService

    private val TAG: String = "MainActivity"

    private val SERVER_ENDPOINT: String =
        "https://us-central1-development-382019.cloudfunctions.net/cloudscan"
    private val BINARY_GRAPH_NAME: String = "feature_extraction_desktop.binarypb"
    private val INPUT_VIDEO_STREAM_NAME: String = "input_video_gpu"
    private val OUTPUT_VIDEO_STREAM_NAME: String = "output_video"
    private val CAMERA_FACING: CameraHelper.CameraFacing = CameraHelper.CameraFacing.BACK
    private val FLIP_FRAMES_VERTICALLY: Boolean = true

    private val httpClient: OkHttpClient =
        OkHttpClient.Builder().connectTimeout(1, TimeUnit.SECONDS).readTimeout(1, TimeUnit.SECONDS)
            .build()
    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private var previewFrameTexture: SurfaceTexture? = null

    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private val previewDisplayView: SurfaceView? = null


    @Volatile
    private var matchFound = false

    @Volatile
    private var processing = false
    private var imgIdx: JSONArray = JSONArray()

    private val processingLock = ReentrantLock()

    // private Packet featuresPacket;
    private var currentFeatsTs: Long = 0
    // Creates and manages an {@link EGLContext}.
    private lateinit var eglManager: EglManager

    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private lateinit var processor: FrameProcessor

    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private lateinit var converter: ExternalTextureConverter

    // Handles camera access via the {@link CameraX} Jetpack support library.
    private val cameraHelper: CameraXPreviewHelper? = null

    private val executorService: ExecutorService = Executors.newFixedThreadPool(4)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.restartScan.setOnClickListener { restartScanning() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        AndroidAssetUtil.initializeNativeAssetManager(this)

        eglManager = EglManager(null)

        Log.d(TAG,"EGL Initialized")
        processor = FrameProcessor(
            this,
            eglManager.nativeContext,
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )
        Log.d(TAG,"Frame Processor Initialized")
        //        PermissionHelper.checkAndRequestCameraPermissions(this);
        processor.graph.addPacketToInputStream(
            "match_image", processor.packetCreator.createRgbImageFrame(
                Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
            ), System.currentTimeMillis()
        )
        Log.d(TAG,"Packet added to MATCH IMAGE STREAM")
        processor.graph.addPacketToInputStream(
            "enable_scanning",
            processor.packetCreator.createBool(true),
            System.currentTimeMillis()
        )
        Log.d(TAG,"Packet added to ENABLE SCANNING STREAM")
        // Add packet callbacks for new outputs
        processor.addPacketCallback(
            "box_floats"
        ) { packet: Packet? ->
            try {
                Log.e(TAG, "Got floats: ")
                val boxFloats = PacketGetter.getFloat32Vector(packet)
                updateView(boxFloats.contentToString())
            } catch (e: java.lang.Exception) {
                Log.e(TAG, "Error getting box floats: " + e.message)
            }
        }
        Log.d(TAG,"BOX FLOATS OP STREAM CALLBACK added")
        processor.addPacketCallback(
            "output_tensor_floats"
        ) { packet: Packet ->
            if (!matchFound && !processing) {
                val embeddingBytes = PacketGetter.getFloat32Vector(packet)
                sendEmbeddingToServer(embeddingBytes)
                processingLock.tryLock()
                currentFeatsTs = packet.timestamp
                processingLock.unlock()
            }
        }
        Log.d(TAG,"OP TENSOR INDEX OP STREAM CALLBACK added")
        processor.addPacketCallback(
            "rr_index"
        ) { packet: Packet? ->
            val index = PacketGetter.getInt32(packet)
            try {
                if (index < imgIdx.length() && !matchFound && imgIdx != null && index >= 0) {
                    executorService.submit(Runnable {
                        processingLock.lock()
                        try {
                            matchFound = true
                            val url = URL(imgIdx.optString(index))
                            val connection = url.openConnection() as HttpURLConnection
                            connection.requestMethod = "GET"
                            connection.doInput = true
                            connection.connect()

                            val inputStream = connection.inputStream
                            val bitmap = BitmapFactory.decodeStream(inputStream)
                            updateView("Image downloaded")
                            val imagePacket = processor.packetCreator.createRgbImageFrame(bitmap)
                            processor.graph.addPacketToInputStream(
                                "match_image", imagePacket,
                                System.currentTimeMillis()
                            )
                            processor.graph.addPacketToInputStream(
                                "enable_scanning",
                                processor.packetCreator.createBool(false),
                                System.currentTimeMillis()
                            )
                            bitmap.recycle()
                        } catch (_: ProtocolException) {
                        } catch (_: IOException) {
                        } finally {
                            processingLock.unlock()
                        }
                    })
                }
            } catch (e: java.lang.Exception) {
                Log.e(TAG, "Error accessing image index: " + e.message)
            }
        }
        Log.d(TAG,"RR INDEX OP STREAM CALLBACK added")
    }

    private fun sendEmbeddingToServer(embeddingBytes: FloatArray?) {
        executorService.submit {
            try {
                processingLock.tryLock()
                if (processing) {
                    updateView("Processing...")
                    processingLock.unlock()
                    return@submit
                }
                processing = true
                processingLock.unlock()
                // Convert float array to byte array
                val byteBuffer =
                    ByteBuffer.allocate(embeddingBytes!!.size * 4) // 4 bytes per float
                byteBuffer.order(ByteOrder.nativeOrder()) // Use the native byte order
                for (value in embeddingBytes) {
                    byteBuffer.putFloat(value)
                }
                val embeddingByteArray = byteBuffer.array()

                // Create the request
                val body: RequestBody = RequestBody.create(
                    MediaType.parse("application/octet-stream"),
                    embeddingByteArray
                )
                val request: Request = Request.Builder()
                    .url(SERVER_ENDPOINT)
                    .post(body)
                    .build()

                // Send the request
                updateView("Detecting...")
                httpClient.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        processingLock.tryLock()
                        processing = false
                        processingLock.unlock()
                        Log.e(
                            TAG,
                            "Error sending embedding to server: " + e.message
                        )
                    }

                    override fun onResponse(call: Call, response: Response) {
                        var images: JSONArray? = null
                        try {
                            if (response.isSuccessful()) {
                                val responseBody: String = response.body().toString()

                                Log.d(TAG,responseBody)
                                val Jobject = JSONObject(responseBody)
                                images = Jobject.getJSONArray("images")
                                if (images.length() == 0) {
                                    Log.e(
                                        TAG,
                                        "No images returned from server"
                                    )
                                    updateView("No detections")
                                    return
                                } else {
                                    updateView("Detection : " + images.length())
                                }

                                val features = Jobject.getJSONArray("features")

                                if (features.length() == 0) {
                                    Log.e(
                                        TAG,
                                        "No features returned from server"
                                    )
                                } else {
                                    processor.graph.addPacketToInputStream(
                                        "query_feats",
                                        createQueryFeaturesPacket(features),
                                        currentFeatsTs
                                    )
                                }
                            } else {
                                Log.e(
                                    TAG,
                                    "Server returned an error: " + response.code()
                                )
                            }
                        } catch (e: JSONException) {
                            Log.e(
                                TAG,
                                "Error reading server response: " + e.message
                            )
                        } catch (e: IOException) {
                            Log.e(
                                TAG,
                                "Error reading server response: " + e.message
                            )
                        } finally {
                            try {
                                processingLock.tryLock()
                                processing = false
                                imgIdx = images!!
                                processingLock.unlock()
                                response.close()
                            } catch (e: java.lang.Exception) {
                                response.close()
                            }
                        }
                    }
                })
            } catch (e: java.lang.Exception) {
                Log.e(TAG, "Error creating JSON payload: " + e.message)
            }
        }
    }

    private fun createQueryFeaturesPacket(features: JSONArray): Packet? {
        if (features == null || features.length() == 0) {
            return null // No features to send
        }


        // Join the array of strings into a single string, separated by a delimiter
        try {
            val joinedFeatures: String = features.join("|")
            return processor.packetCreator.createString(joinedFeatures)
        } catch (e: JSONException) {
            Log.e(TAG, "Error creating query features packet: " + e.message)
            return null
        }
    }

    private fun updateView(s: String) {
        val toast = Toast.makeText(this@MainActivity,s,Toast.LENGTH_SHORT)
        toast.show()
    }

    private fun restartScanning() {
        Log.d(TAG,"SCANNING RESTARTED")
        processingLock.tryLock()
        matchFound = false
        processing = false
        // imgIdx = null;
        processor.graph.addPacketToInputStream(
            "enable_scanning",
            processor.packetCreator.createBool(true),
            System.currentTimeMillis()
        )
        processingLock.unlock()
        updateView("Detection restarted") // Update the UI
    }

    private fun captureVideo() {}

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        )
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(
                    baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                startCamera()
            }
        }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    init {
        System.loadLibrary("mediapipe_jni")
        System.loadLibrary("opencv_java4")
    }

    companion object {
        // Load all native libraries needed by the app.
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    override fun onResume() {
        super.onResume()
        converter = ExternalTextureConverter(eglManager.context,
            2)
        converter.setFlipY(FLIP_FRAMES_VERTICALLY)
        converter.setConsumer(processor)
        processor
            .videoSurfaceOutput
            .setFlipY(
                FLIP_FRAMES_VERTICALLY
            )
    }

    override fun onPause() {
        super.onPause()
        converter.close()
    }
}
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.injini.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.injini.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 3
        versionName = "0.3"
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // The embedder .onnx is stored raw so ONNX Runtime maps it directly.
    androidResources { noCompress += "onnx" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.cardview:cardview:1.0.0")

    // Same .onnx pipeline as SiloSense: XNNPACK runs Arm NEON kernels, session
    // profiling reports which provider actually executed each node.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // Real multipart file upload for HfSync — hand-rolled multipart boundary
    // strings over HttpURLConnection are easy to get subtly wrong for
    // something that uploads real training data; OkHttp's builder is not.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    testImplementation("junit:junit:4.13.2")
}

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');
const logOutput = document.getElementById('log-output');

// Capture button element
const captureBtn = document.getElementById('captureBtn');
if (captureBtn) captureBtn.disabled = true;

let currentUser = null;
let currentObjects = [];

import { FilesetResolver, ObjectDetector } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2';

let mpDetector;
uiLog('🔄 Initializing MediaPipe Object Detector...');
(async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm'
  );
  mpDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    scoreThreshold: 0.5,
    maxResults: 5
  });
  uiLog('✅ MediaPipe Object Detector initialized');
})();

function uiLog(message) {
  if (logOutput) logOutput.innerText += message + '\n';
  console.log(message);
}

let faceMatcher;

uiLog("🔄 Loading models...");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('./models/tiny_face_detector'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('./models')
]).then(async ([, ,]) => {
  uiLog("✅ All models loaded");

  const descriptors = [];
  for (let i = 1; i <= 16; i++) {
    const imgPath = `labeled_images/arun/arun${i}.jpg`;
    uiLog(`🔍 Loading image: ${imgPath}`);

    try {
      const img = await faceapi.fetchImage(imgPath);
      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        uiLog(`✅ Face detected in arun${i}.jpg`);
        descriptors.push(detection.descriptor);
      } else {
        uiLog(`⚠️ No face detected in arun${i}.jpg`);
      }
    } catch (err) {
      uiLog(`❌ Error processing ${imgPath}: ${err.message}`);
    }
  }

  if (descriptors.length === 0) {
    uiLog("❌ No face descriptors loaded. Aborting.");
    return;
  }

  faceMatcher = new faceapi.FaceMatcher(
    [new faceapi.LabeledFaceDescriptors("Arun", descriptors)],
    0.6
  );

  uiLog("✅ Face matcher initialized");
  startVideo();
}).catch(err => {
  console.error("❌ Model load error:", err);
  uiLog("❌ Model load error: " + err.message);
});

function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        uiLog("🎥 Camera stream started");
        detectLoop();
      };
    })
    .catch((err) => {
      console.error("❌ Camera error: ", err);
      alert('Camera access denied or failed.');
    });
}

async function detectLoop() {
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(overlay, displaySize);

  setInterval(async () => {
    canvas.clearRect(0, 0, overlay.width, overlay.height);

    // Face detection
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions()
    ).withFaceLandmarks().withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    resizedDetections.forEach(detection => {
      const box = detection.detection.box;
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      new faceapi.draw.DrawBox(
        { x: box.x, y: box.y, width: box.width, height: box.height },
        { label: bestMatch.toString() }
      ).draw(overlay);
      uiLog(`✅ Face detected: ${bestMatch.toString()}`);
    });

    // MediaPipe Object detection (cyan boxes)
    if (mpDetector) {
      const mpResult = await mpDetector.detectForVideo(video, performance.now());
      mpResult.detections.forEach(d => {
        const box = d.boundingBox;
        const [x, y, w, h] = [box.originX, box.originY, box.width, box.height];
        canvas.strokeStyle = 'cyan';
        canvas.lineWidth = 2;
        canvas.strokeRect(x, y, w, h);
        canvas.font = '16px Arial';
        canvas.fillStyle = 'cyan';
        const category = d.categories[0];
        const label = `${category.categoryName} (${(category.score * 100).toFixed(1)}%)`;
        canvas.fillText(label, x, y > 10 ? y - 5 : 10);
        uiLog(`🟦 MP Detected: ${label}`);
      });

      // Only count non-person objects for storing
      const nonPersonDetections = mpResult.detections.filter(d => d.categories[0].categoryName !== 'person');
      const faceCount = resizedDetections.length;
      const itemCount = nonPersonDetections.length;
      if (faceCount === 1 && itemCount > 0) {
        currentUser = faceMatcher.findBestMatch(resizedDetections[0].descriptor).label;
        currentObjects = nonPersonDetections.map(d => d.categories[0].categoryName);
      } else {
        currentUser = null;
        currentObjects = [];
      }

      // Update capture button enabled state
      if (captureBtn) {
        captureBtn.disabled = !(currentUser && currentObjects.length > 0);
        // Visual feedback for button state
        captureBtn.style.opacity = captureBtn.disabled ? '0.5' : '1.0';
        uiLog(`🖱️ Capture button ${captureBtn.disabled ? 'disabled' : 'enabled'}`);
      }
    }
  }, 500);
}

if (captureBtn) {
  captureBtn.addEventListener('click', async () => {
    uiLog(`👤 currentUser: ${currentUser}, 📦 currentObjects: ${currentObjects.join(', ')}`);
    if (!currentUser || currentObjects.length === 0) {
      uiLog('⚠️ No user or objects detected to capture.');
      return;
    }

    // Create a temporary canvas to capture the current video frame
    const snapshotCanvas = document.createElement('canvas');
    snapshotCanvas.width = video.videoWidth;
    snapshotCanvas.height = video.videoHeight;
    const snapshotCtx = snapshotCanvas.getContext('2d');
    snapshotCtx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    // Get image data as base64
    const imageData = snapshotCanvas.toDataURL('image/png');

    const payload = {
      user: currentUser,
      objects: currentObjects,
      snapshot: imageData
    };

    uiLog('📤 Sending snapshot and data to server...');

    try {
      const response = await fetch('/api/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        uiLog('✅ Snapshot and data sent successfully.');
      } else {
        uiLog(`❌ Server responded with status: ${response.status}`);
      }
    } catch (err) {
      uiLog(`❌ Error sending data: ${err.message}`);
    }
  });
}
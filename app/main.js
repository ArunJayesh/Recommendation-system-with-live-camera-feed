const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');
const logOutput = document.getElementById('log-output');

// Capture button element
const captureBtn = document.getElementById('captureBtn');
if (captureBtn) captureBtn.disabled = true;

// Recommendation button element
const recommendBtn = document.getElementById('recommendBtn');
if (recommendBtn) {
  recommendBtn.disabled = true;
  recommendBtn.style.opacity = '0.5';
}

let currentUser = null;
let currentObjects = [];
// persisted user→items mapping
let userItems = {};


// ----------------------
// Recommendation logic
// ----------------------
const RECOMMENDATION_MAP = {
  "cell phone":    ["phone case", "screen protector", "wireless charger"],
  "bottle":   ["bottle brush", "insulated sleeve", "cleaning tablets"],
  "laptop": ["laptop stand", "micro fiber cloth", "mouse and keyboard"],
  "chair": ["table", "cleaning towel", "back cushion"],
  // …etc
};

function getRecommendationsFor(items) {
  const recs = new Set();
  items.forEach(item => {
    (RECOMMENDATION_MAP[item] || []).forEach(r => recs.add(r));
  });
  return Array.from(recs);
}

import { FilesetResolver, ObjectDetector } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2';

let mpDetector;
uiLog('🔄 Initializing MediaPipe Object Detector...');
// Load FaceAPI models before running any detections
(async () => {
  uiLog("🔄 Loading FaceAPI models...");
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('models')
  ]);
  uiLog("✅ FaceAPI models loaded");
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

uiLog("🔄 Loading FaceAPI models for recognition...");
await Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('models')
]);
uiLog("✅ FaceAPI models loaded for recognition");
// Dynamically load labeled faces from each subfolder in labeled_images
const labels = ['arun', 'pranav']; // Add new folder names here as you add users
const labeledDescriptors = [];

(async () => {
  for (const label of labels) {
    const descriptors = [];
    for (let i = 1;; i++) {
      const imgPath = `labeled_images/${label}/${label}${i}.jpg`;
      uiLog(`🔍 Loading image: ${imgPath}`);
      try {
        const img = await faceapi.fetchImage(imgPath);
        const detection = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (detection) {
          uiLog(`✅ Face detected in ${label}${i}.jpg`);
          descriptors.push(detection.descriptor);
        } else {
          uiLog(`⚠️ No face detected in ${label}${i}.jpg, stopping for ${label}.`);
          break;
        }
      } catch {
        // no more images for this label
        break;
      }
    }
    if (descriptors.length > 0) {
      // Capitalize label for display
      const displayLabel = label.charAt(0).toUpperCase() + label.slice(1);
      labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(displayLabel, descriptors));
    }
  }

  if (labeledDescriptors.length === 0) {
    uiLog("❌ No face descriptors loaded. Aborting.");
    return;
  }

  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  uiLog("✅ Face matcher initialized");
  startVideo();
})().catch(err => {
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

    // draw each detection, highlighting persisted items in red
    if (mpDetector) {
      const mpResult = await mpDetector.detectForVideo(video, performance.now());
      mpResult.detections.forEach(d => {
        const box = d.boundingBox;
        const [x, y, w, h] = [box.originX, box.originY, box.width, box.height];
        const category = d.categories[0];
        const itemName = category.categoryName;
        const score = (category.score * 100).toFixed(1);
        // determine the current face count and label
        const faceCount = resizedDetections.length;
        let faceLabel = null;
        if (faceCount === 1) {
          faceLabel = faceMatcher.findBestMatch(resizedDetections[0].descriptor).label;
        }
        // check if this item was previously captured for this user
        const isPersisted = faceLabel && userItems[faceLabel] && userItems[faceLabel].includes(itemName);
        // choose color and label
        const color = isPersisted ? 'red' : 'cyan';
        const displayLabel = isPersisted ? `${faceLabel}'s ${itemName}` : `${itemName} (${score}%)`;
        // draw bounding box
        canvas.strokeStyle = color;
        canvas.lineWidth = 2;
        canvas.strokeRect(x, y, w, h);
        // draw text
        canvas.font = '16px Arial';
        canvas.fillStyle = color;
        canvas.fillText(displayLabel, x, y > 10 ? y - 5 : 10);
        // log
        uiLog(`${isPersisted ? '🔴' : '🟦'} MP Detected: ${displayLabel}`);
      });

      // Determine current user from face detection
      const faceCount = resizedDetections.length;
      if (faceCount === 1) {
        currentUser = faceMatcher.findBestMatch(resizedDetections[0].descriptor).label;
      } else {
        currentUser = null;
      }

      // Determine current objects separately
      const nonPersonDetections = mpResult.detections.filter(d => d.categories[0].categoryName !== 'person');
      if (currentUser && nonPersonDetections.length > 0) {
        currentObjects = nonPersonDetections.map(d => d.categories[0].categoryName);
      } else {
        currentObjects = [];
      }

      // Update capture button enabled state: only when exactly 1 user and ≥1 objects
      const hasExactlyOneUser = faceCount === 1;
      const hasObjects = currentObjects.length > 0;
      if (captureBtn) {
        captureBtn.disabled = !(hasExactlyOneUser && hasObjects);
        captureBtn.style.opacity = captureBtn.disabled ? '0.5' : '1.0';
        uiLog(`🖱️ Capture button ${captureBtn.disabled ? 'disabled' : 'enabled'}`);
      }
      // Update recommendation button enabled state
      if (recommendBtn) {
        const hasPersisted = currentUser && userItems[currentUser] && userItems[currentUser].length > 0;
        recommendBtn.disabled = !hasPersisted;
        recommendBtn.style.opacity = recommendBtn.disabled ? '0.5' : '1.0';
        uiLog(`🖱️ Recommendation button ${recommendBtn.disabled ? 'disabled' : 'enabled'}`);
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
      const response = await fetch('/api/captures', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        uiLog('✅ Snapshot and data sent successfully.');
      } else {
        uiLog(`❌ Server responded with status: ${response.status}`);
      }
      // persist this user’s captured items until reload
      userItems[currentUser] = [...currentObjects];
      uiLog(`🔖 Persisted items for ${currentUser}: ${userItems[currentUser].join(', ')}`);
    } catch (err) {
      uiLog(`❌ Error sending data: ${err.message}`);
    }
  });
}

// Recommendation button handler
if (recommendBtn) {
  recommendBtn.addEventListener('click', () => {
    if (!currentUser) {
      uiLog('⚠️ No user selected for recommendations.');
      return;
    }
    const captured = userItems[currentUser] || [];
    const recommendations = getRecommendationsFor(captured);
    const recContainer = document.getElementById('recommendations');
    if (recContainer) {
      recContainer.innerHTML = `
        <h4>Recommended for ${currentUser}:</h4>
        <ul>
          ${recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      `;
    }
    uiLog(`💡 Recommendations for ${currentUser}: ${recommendations.join(', ')}`);
  });
}
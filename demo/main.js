const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');
const logOutput = document.getElementById('log-output');

function uiLog(message) {
  if (logOutput) logOutput.innerText += message + '\n';
  console.log(message);
}

let faceMatcher;
let cocoModel;

uiLog("🔄 Loading models...");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('../face_detection/models/tiny_face_detector'),
  faceapi.nets.faceLandmark68Net.loadFromUri('../face_detection/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('../face_detection/models'),
  cocoSsd.load()
]).then(async ([, , , loadedCocoModel]) => {
  uiLog("✅ All models loaded");
  cocoModel = loadedCocoModel;

  const descriptors = [];
  for (let i = 1; i <= 16; i++) {
    const imgPath = `../face_detection/labeled_images/arun/arun${i}.jpg`;
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

    // Object detection
    const predictions = await cocoModel.detect(video);
    predictions.forEach(pred => {
      const [x, y, width, height] = pred.bbox;
      canvas.strokeStyle = 'red';
      canvas.lineWidth = 2;
      canvas.strokeRect(x, y, width, height);
      canvas.font = '16px Arial';
      canvas.fillStyle = 'red';
      canvas.fillText(
        `${pred.class} (${(pred.score * 100).toFixed(1)}%)`,
        x,
        y > 10 ? y - 5 : 10
      );
      uiLog(`📦 Detected object: ${pred.class} (${(pred.score * 100).toFixed(1)}%)`);
    });
  }, 500);
}

document.getElementById('captureBtn').addEventListener('click', (e) => {
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const snapCtx = snapCanvas.getContext('2d');
  snapCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
  console.log("📸 Snapshot captured");
  e.target.style.backgroundColor = '#e74c3c';
});
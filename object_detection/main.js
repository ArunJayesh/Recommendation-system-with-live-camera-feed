


const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');
const logOutput = document.getElementById('log-output');

let model;

// Load the COCO-SSD model
cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  console.log("✅ COCO-SSD model loaded successfully");
  startVideo();
}).catch((err) => {
  console.error("❌ Error loading COCO-SSD model:", err);
});

// Start webcam video
function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        detectFrame();
      };
      console.log("✅ Camera stream started");
    })
    .catch((err) => {
      console.error("❌ Camera access error:", err);
    });
}

// Run object detection on each frame
function detectFrame() {
  model.detect(video).then(predictions => {
    canvas.clearRect(0, 0, overlay.width, overlay.height);
    predictions.forEach(pred => {
      drawBoundingBox(pred);
      logDetection(pred);
    });
    requestAnimationFrame(detectFrame);
  });
}

// Draw bounding box
function drawBoundingBox(prediction) {
  const [x, y, width, height] = prediction.bbox;
  canvas.strokeStyle = 'yellow';
  canvas.lineWidth = 2;
  canvas.strokeRect(x, y, width, height);
  canvas.font = '16px Arial';
  canvas.fillStyle = 'yellow';
  canvas.fillText(`${prediction.class} (${(prediction.score * 100).toFixed(1)}%)`, x, y > 10 ? y - 5 : 10);
}

// Log to UI panel
function logDetection(prediction) {
  const message = `🟡 ${prediction.class} detected (${(prediction.score * 100).toFixed(1)}%)`;
  console.log(message);
  if (logOutput) {
    logOutput.innerText += message + '\n';
  }
}
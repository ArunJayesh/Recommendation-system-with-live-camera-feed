const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');

// Load the face-api models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models') // optional
]).then(startVideo);

function startVideo() {
  // This will ask for camera permissions
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      // If successful, we assign the stream to the video element
      video.srcObject = stream;
    })
    .catch((err) => {
      console.error("Camera error: ", err);
      alert('Camera access denied or failed. Please allow camera permissions.');
    });
}

// Wait for the video metadata to be loaded before starting face recognition
video.addEventListener('loadedmetadata', () => {
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  video.width = video.parentElement.clientWidth;
  video.height = video.parentElement.clientHeight;
  overlay.width = displaySize.width;
  overlay.height = displaySize.height;
  faceapi.matchDimensions(overlay, displaySize);

  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions()
    ).withFaceLandmarks().withFaceDescriptors();

    canvas.clearRect(0, 0, overlay.width, overlay.height);
    const resized = faceapi.resizeResults(detections, displaySize);

    resized.forEach((detection) => {
      const box = detection.detection.box;
      const drawBox = new faceapi.draw.DrawBox(
        { x: box.x, y: box.y, width: box.width, height: box.height },
        { label: "Person" }
      );
      drawBox.draw(overlay);
    });
  }, 500);
});

// Change button color when clicked
document.getElementById('captureBtn').addEventListener('click', (e) => {
  // Capture snapshot
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const snapCtx = snapCanvas.getContext('2d');
  snapCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
  const imageData = snapCanvas.toDataURL('image/png');
  
  // Log the captured snapshot
  console.log("Snapshot captured", imageData);
  
  // Change button color when clicked
  e.target.style.backgroundColor = '#e74c3c';  // Example: Change to red
});
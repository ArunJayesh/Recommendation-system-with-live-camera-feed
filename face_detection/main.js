const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const canvas = overlay.getContext('2d');

let faceMatcher;

let personToObjects = {}; // Object to store objects detected per person

let objectDescriptors = [];

const logContainer = document.getElementById('object-info');

// Function to append log messages to the UI
function appendLog(message) {
  // Only append messages that contain '✅' or '🔍', exclude the running and object detection messages
  if (message.includes('Running detection loop') || message.includes('Trying to detect object')) return;
  if (message.includes('✅') || message.includes('🔍')) {
    const p = document.createElement('p');
    p.style.color = 'green';
    p.textContent = message;
    logContainer.appendChild(p);
  }
}

// Function to update object info in the UI
function updateObjectInfo(user, objects) {
  const objectInfoContainer = document.getElementById('object-info');
  objectInfoContainer.innerHTML = `<strong style="color: green;">User detected: ${user}</strong><br>`;

  objects.forEach((obj, index) => {
    objectInfoContainer.innerHTML += `${index + 1}. ${obj}<br>`;
  });
}

// Load labeled images for face recognition
async function loadLabeledImages() {
  const labels = ['Arun']; // Adjust the labels as needed
  return Promise.all(
    labels.map(async (label) => {
      const descriptors = [];
      const labelUpper = label.toUpperCase();
      
      for (let i = 1; i <= 2; i++) {
        appendLog(`🔍 Loading image: labeled_images/${labelUpper}/${labelUpper}${i}.jpg`);
        const img = await faceapi.fetchImage(`labeled_images/${labelUpper}/${labelUpper}${i}.jpg`);
        
        try {
          const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          
          if (detection) {
            appendLog(`✅ Face detected in ${labelUpper}${i}.jpg`);
            descriptors.push(detection.descriptor);
          } else {
            console.warn(`No face detected for ${labelUpper} image ${i}`);
          }
        } catch (err) {
          console.error(`Error loading image labeled_images/${labelUpper}/${labelUpper}${i}.jpg:`, err);
        }
      }
      
      return new faceapi.LabeledFaceDescriptors(label, descriptors);
    })
  );
}

async function loadLabeledObjects() {
  const objectLabels = ['black_phone', 'blue_bottle', 'white_earbuds'];
  const descriptors = [];

  for (const label of objectLabels) {
    for (let i = 1; i <= 6; i++) {
      const imagePath = `labeled_object/${label}/${label}${i}.jpg`;
      try {
        const img = await faceapi.fetchImage(imagePath);
        const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        if (detection) {
          descriptors.push({ label, descriptor: detection.descriptor });
          appendLog(`✅ Loaded object: ${label}${i}`);
        }
      } catch (err) {
        console.warn(`⚠️ Could not load object image: ${imagePath}`);
        continue; // Skip to next
      }
    }
  }

  return descriptors;
}

function classifyObject(descriptor) {
  if (!objectDescriptors.length) return null;

  const best = objectDescriptors.reduce((bestMatch, obj) => {
    const dist = faceapi.euclideanDistance(descriptor, obj.descriptor);
    return dist < bestMatch.distance ? { label: obj.label, distance: dist } : bestMatch;
  }, { label: null, distance: Infinity });

  return best.distance < 0.6 ? best.label : null;
}

// Load models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('models'),
  faceapi.nets.faceExpressionNet.loadFromUri('models') // Optional
]).then(async () => {
  const labeledDescriptors = await loadLabeledImages();
  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  objectDescriptors = await loadLabeledObjects();
  startVideo();
});

// Start video stream
function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      console.log("✅ Camera stream started");
    })
    .catch((err) => {
      console.error("❌ Camera error: ", err);
      alert('Camera access denied or failed. Please allow camera permissions.');
    });
}

// Video and face recognition initialization
video.addEventListener('loadeddata', () => {
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  video.width = video.parentElement.clientWidth;
  video.height = video.parentElement.clientHeight;
  overlay.width = video.width;
  overlay.height = video.height;
  faceapi.matchDimensions(overlay, displaySize);

  console.log("📷 Video loaded. Starting detection loop.");
  console.log("Video size:", video.width, video.height);
  console.log("Overlay size:", overlay.width, overlay.height);

  setInterval(async () => {
    console.log("🔍 Running detection loop");
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions()
    ).withFaceLandmarks().withFaceDescriptors();

    canvas.clearRect(0, 0, overlay.width, overlay.height);
    const resized = faceapi.resizeResults(detections, displaySize);

    resized.forEach((detection) => {
      const box = detection.detection.box;
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      const drawBox = new faceapi.draw.DrawBox(
        { x: box.x, y: box.y, width: box.width, height: box.height },
        { label: bestMatch.toString() }
      );
      drawBox.draw(overlay);

      // Detect objects after face detection
      // (Removed detectObjectsAndUpdateUI as coco-ssd is no longer used)

      // Try classifying an object using face descriptor (as placeholder)
      console.log(`🎯 Trying to detect object...`);
      const objectLabel = classifyObject(detection.descriptor);
      if (objectLabel) {
        console.log(`🟡 Object detected: ${objectLabel}`);
        console.log(`✅ Object assigned to ${bestMatch.label}`);
        canvas.strokeStyle = 'yellow';
        canvas.lineWidth = 2;
        canvas.strokeRect(box.x, box.y, box.width, box.height);
        canvas.font = '16px Arial';
        canvas.fillStyle = 'yellow';
        canvas.fillText(objectLabel, box.x, box.y > 10 ? box.y - 5 : 10);
        if (!personToObjects[bestMatch.label].includes(objectLabel)) {
          personToObjects[bestMatch.label].push(objectLabel);
          updateObjectInfo(bestMatch.label, personToObjects[bestMatch.label]);
        }
      }
    });
  }, 500);
});

// Capture snapshot
document.getElementById('captureBtn').addEventListener('click', (e) => {
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const snapCtx = snapCanvas.getContext('2d');
  snapCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
  const imageData = snapCanvas.toDataURL('image/png');
  
  console.log("Snapshot captured", imageData);
  e.target.style.backgroundColor = '#e74c3c';  // Change button color to red
});
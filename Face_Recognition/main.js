import * as faceapi from 'face-api.min.js';
const video = document.getElementById("video");
const isScreenSmall = window.matchMedia("(max-width: 700px)");
let predictedAges = [];

/****Loading the model ****/
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models/"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models/"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models/"),
  faceapi.nets.faceExpressionNet.loadFromUri("/models/"),
  faceapi.nets.ageGenderNet.loadFromUri("/models/")
]).then(startVideo);

function startVideo() {
  alert("problem");
  navigator.getUserMedia(
    { video: {} },
    stream => (video.srcObject = stream),
    err => console.error(err)
  );
}
function getLabeledFaceDescriptions() {
  const labels = ["Sanjiv", "Ratan", "Data"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
       
       
        console.log(`hello ${label}`)
      }
     
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
      
    })
  );
}

video.addEventListener("play", async()=>{
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  
    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);

    const displaySize = {width: video.width, height: video.height};
faceapi.matchDimensions(canvas, displaySize);

setInterval(async ()=>{
  const detections = await faceapi
  .detectAllFaces(video)
  .withFaceLandmarks()
  .withFaceDescriptors();

  const resizedDetections = faceapi.resizeResults(detections, displaySize)
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

  const results = resizedDetections.map((d) => {
    return faceMatcher.findBestMatch(d.descriptor);
  });
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: result,
    });
    drawBox.draw(canvas);
   
  });
}, 100);
});

/****Fixing the video with based on size size  ****/
function screenResize(isScreenSmall) {
  if (isScreenSmall.matches) {
    video.style.width = "320px";
  } else {
    video.style.width = "500px";
  }
}

screenResize(isScreenSmall);
isScreenSmall.addListener(screenResize);

/****Event Listeiner for the video****/
video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  let container = document.querySelector(".container");
  container.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    /****Drawing the detection box and landmarkes on canvas****/
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

    /****Setting values to the DOM****/
    if (resizedDetections && Object.keys(resizedDetections).length > 0) {
      const age = resizedDetections.age;
      const interpolatedAge = interpolateAgePredictions(age);
      const gender = resizedDetections.gender;
      const expressions = resizedDetections.expressions;
      const maxValue = Math.max(...Object.values(expressions));
      const emotion = Object.keys(expressions).filter(
        item => expressions[item] === maxValue
      );
      document.getElementById("age").innerText = `This man is getting old - ${interpolatedAge}`;
      document.getElementById("gender").innerText = `This man is having sex - ${gender}`;
      document.getElementById("emotion").innerText = `This man's Expression is - ${emotion[0]}`;
    }
  }, 10);
});

function interpolateAgePredictions(age) {
  predictedAges = [age].concat(predictedAges).slice(0, 30);
  const avgPredictedAge =
    predictedAges.reduce((total, a) => total + a) / predictedAges.length;
  return avgPredictedAge;
}
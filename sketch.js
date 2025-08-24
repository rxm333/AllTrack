let video;

// Hands var
let handPose;
let hands = [];
let connections;
let optionsHands ={
  maxHands: 2,
  flipped: true,
  runtime: "mediapipe",
  modelType: "full",
  detectorModelUrl: undefined, //default to use the tf.hub model
  landmarkModelUrl: undefined, //default to use the tf.hub model
};

// Faces var
let faceMesh;
let faces = [];
let optionsFaces = { 
  maxFaces: 1, 
  refineLandmarks: true, 
  flipped: true 
};
let triangles;
let leftEyeIris = [468,469,470,471,472];
let rightEyeIris = [473,474,475,476,477];

// Body var
let bodyPose;
let poses = [];
let connectionsPoses;
let optionsBodies={
  modelType: "MULTIPOSE_LIGHTNING", // "MULTIPOSE_LIGHTNING", "SINGLEPOSE_LIGHTNING", or "SINGLEPOSE_THUNDER".
  enableSmoothing: true,
  minPoseScore: 0.25,
  multiPoseMaxDimension: 256,
  enableTracking: true,
  trackerType: "boundingBox", // "keypoint" or "boundingBox"
  trackerConfig: {},
  modelUrl: undefined,
  flipped: true
}

function preload(){
  handPose = ml5.handPose(optionsHands);
  faceMesh = ml5.faceMesh(optionsFaces);
  bodyPose = ml5.bodyPose(optionsBodies);


}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO, { flipped: true });
  video.size(640, 480);
  video.hide();

  // Hands
  handPose.detectStart(video, gotHands);
  connections = handPose.getConnections();

  // Faces
  faceMesh.detectStart(video, gotFaces);
  triangles = faceMesh.getTriangles();

  // Body
  bodyPose.detectStart(video, gotPoses);
  connectionsPoses = bodyPose.getSkeleton();
  
}

function draw() {
  background(0);
  // image(video, 0, 0, width, height);

  // HANDS
  for (let hand of hands) {
    drawHandSkeleton(hand.keypoints, connections, "blue", 2);
    drawHandPointNumbers(hand.keypoints, "yellow", 12, 4);
    drawHandPoints(hand.keypoints, "magenta", 5);
  }

  // Helper functions for hands
  function drawHandPoints(handKeypoints, color, size) {
    fill(color);
    noStroke();
    for (let keypoint of handKeypoints) {
      circle(keypoint.x, keypoint.y, size);
    }
  }

  function drawHandSkeleton(handKeypoints, handConnections, color, weight) {
    stroke(color);
    strokeWeight(weight);
    for (let [pointAIndex, pointBIndex] of handConnections) {
      let pointA = handKeypoints[pointAIndex];
      let pointB = handKeypoints[pointBIndex];
      line(pointA.x, pointA.y, pointB.x, pointB.y);
    }
  }

  function drawHandPointNumbers(handKeypoints, color, size, weight) {
    fill(color);
    noStroke();
    textSize(size);
    textAlign('center');
    for (let i = 0; i < handKeypoints.length; i++) {
      text(i, handKeypoints[i].x, handKeypoints[i].y);
    }
  }

  // FACES
  for (let face of faces){
    drawFacePoints(face.keypoints, "magenta", 4);
    drawFaceTriangles(face.keypoints, "blue", 1);
    drawFacePointNumbers(face.keypoints, "magenta", 12, 4);
    drawIrisPoints(face.keypoints, leftEyeIris, "yellow", 4);
    drawIrisPoints(face.keypoints, rightEyeIris, "yellow", 4);

  }

  // Helper functions for faces
  function drawFacePoints(faceKeypoints, color, size) {
    fill(color);
    noStroke()
    for (let keypoint of faceKeypoints) {
      circle(keypoint.x, keypoint.y, size);
    }
  }

  function drawFacePointNumbers(faceKeypoints, color, size, weight) {
    fill(color);
    noStroke();
    textSize(size);
    textAlign('center');
    for (let i =0; i < faceKeypoints.length; i++) {
      text(i, faceKeypoints[i].x, faceKeypoints[i].y);
    }
  }

  function drawFaceTriangles(faceKeypoints, color, weight) {
    for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    for (let j = 0; j < triangles.length; j++) {
      let indices = triangles[j];
      let pointAIndex = indices[0];
      let pointBIndex = indices[1];
      let pointCIndex = indices[2];
      let pointA = face.keypoints[pointAIndex];
      let pointB = face.keypoints[pointBIndex];
      let pointC = face.keypoints[pointCIndex];

      noFill();
      stroke(color);
      strokeWeight(weight);
      triangle(pointA.x, pointA.y, pointB.x, pointB.y, pointC.x, pointC.y); 
      } 
    }
  }

  function drawIrisPoints(faceKeypoints, irisIndices, color, size) {
    fill(color);
    noStroke();
    for (let index of irisIndices) {
      let point = faceKeypoints[index];
      if (point) {
        circle(point.x, point.y, size);
      }
    }
  }

  // Bodies
  for (let pose of poses){
    drawBodySkeleton(pose.keypoints, "lime", 10);
  }

  function drawBodySkeleton(poseKeypoints, color, weight) {
    stroke(color);
    strokeWeight(weight);
    noFill();
    for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
        for (let j = 4; j < connectionsPoses.length; j++) {
          let pointAIndex = connectionsPoses[j][0];
          let pointBIndex = connectionsPoses[j][1];
          let pointA = pose.keypoints[pointAIndex];
          let pointB = pose.keypoints[pointBIndex];
            if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
              line(pointA.x, pointA.y, pointB.x, pointB.y);
      }
    }
  }


    }
}


function gotHands(resultsHands) {
  hands = resultsHands;
}

function gotFaces(resultsFaces) {
  faces = resultsFaces;
}
function gotPoses(resultsPoses) {
  poses = resultsPoses;
}

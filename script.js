const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}
function customPrint(line) {
  let p = document.createElement('p');
  p.innerText = line;
  document.body.appendChild(p)
}
function imageFileToImageElement(imageFile){
    return new Promise((resolve,reject)=>{
        const imageElement = new Image(640,360);
        imageElement.src = URL.createObjectURL(imageFile);
        imageElement.onload = () => {
            URL.revokeObjectURL(imageElement.src);
            let imageFeatures = tf.tidy(function() {
              imageFrameAsTensor = tf.browser.fromPixels(imageElement);
              let resizedTensorFrame = tf.image.resizeBilinear(imageFrameAsTensor, [input_height, 
        
                  input_width], true);
        
              let normalizedTensorFrame = resizedTensorFrame.div(255);
        
              return mobileNetBase.predict(normalizedTensorFrame.expandDims()).squeeze();
              })
            resolve(imageFeatures)
        }
        imageElement.onerror = (error) =>{
            URL.revokeObjectURL(imageElement.src);
            reject(error)
        }
    })
}

const input_width = 224;
const input_height = 224;
var mobilenet = undefined
var model = null;
var mobileNetBase = undefined
const bioringInput = document.querySelectorAll('#bioring-input');
const otherInput = document.querySelectorAll('#other-input');
const imageInput = document.querySelectorAll('#image-input');


const bioringDataInput = [];
const bioringDataOutput = [];
const imageDataInput = []

const trainButton = document.getElementById('trainButton');
const predictButton = document.getElementById('predictButton');

trainButton.addEventListener('click', function(){
 train();
})
predictButton.addEventListener('click', function(){
  predict();
 })
async function loadMobileNetFeatureModel(){
  const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';
  mobilenet = await tf.loadLayersModel(URL);

  status.innerText = 'MobileNet v2 loaded successfully!';


  const layer = mobilenet.getLayer('global_average_pooling2d_1');

  mobileNetBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output}); 

  mobileNetBase.summary(null, null, customPrint);
  
}

loadMobileNetFeatureModel()

  function train(){
 model = tf.sequential();

model.add(tf.layers.dense({inputShape: [1280], units: 64, activation: 'relu'}));

model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.

model.compile({

    // Adam changes the learning rate over time which is useful.
  
    optimizer: 'adam',
  
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  
    // Else categoricalCrossentropy is used if more than 2 classes.
  
    loss: 'binaryCrossentropy', 
  
    // As this is a classification problem you can record accuracy in the logs too!
  
    metrics: ['accuracy']  
  
  });

  for(let i = 0; i < bioringInput[0].files.length; i++){
        async function f1(){
        const x = await imageFileToImageElement(bioringInput[0].files[i]);
        return x
      }
     bioringDataInput.push(f1())
    bioringDataOutput.push(0);

  }
  
  for(let i = 0; i < otherInput[0].files.length; i++){
        async function f1(){
        const x = await imageFileToImageElement(otherInput[0].files[i]);
        return x
      }
     bioringDataInput.push(f1())
    bioringDataOutput.push(1);
  }
Promise.all(bioringDataInput).then((values)=>{
  
  tf.util.shuffleCombo(values, bioringDataOutput);

  let outputsAsTensor = tf.tensor1d(bioringDataOutput, 'int32');
  let oneHotOutputs = tf.oneHot(outputsAsTensor, 2);
  let inputsAsTensor = tf.stack(values);

  async function f2(){
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 5 });
    status.innerText = 'Model trained successfully!';
      // Make combined model for download.

  let combinedModel = tf.sequential();

  combinedModel.add(mobileNetBase);

  combinedModel.add(model);

  

  combinedModel.compile({

    optimizer: 'adam',

    loss:  'binaryCrossentropy'

  });

  

  combinedModel.summary(null, null, customPrint);

  await combinedModel.save('downloads://my-model');
    outputsAsTensor.dispose();

    oneHotOutputs.dispose();

    inputsAsTensor.dispose();

  }

 f2()
  



})

}


function predict (){
  for(let i = 0; i < imageInput[0].files.length; i++){
    async function f1(){
    const x = await imageFileToImageElement(imageInput[0].files[i]);
    return x
  }
 imageDataInput.push(f1())
}
let bArray = ['bioring', 'not bioring']
Promise.all(imageDataInput).then((values)=>{
  values.forEach((item)=>{
    tf.tidy(()=>{
    let x = item.expandDims([-1])
    let prediction = model.predict(x);
    
    const values = prediction.squeeze().dataSync();
    let cArray = Array.from(values)
    let newDiv = document.getElementById('prediction');
    
    newDiv.innerText = `Prediction: ${bArray[cArray.indexOf(Math.max(...cArray))]} with ${Math.max(...cArray).toFixed(2)*100} % confidence`
  })
  })
    
  
})

}

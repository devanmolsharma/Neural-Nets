document.getElementById('addButton').addEventListener('click', function () {
    // Clone the layer template
    var layerClone = document.querySelector('.card').cloneNode(true);

    // Remove the "show" class to hide the collapsed content
    layerClone.querySelector('.collapse').classList.remove('show');

    // Update the input field to show the output shape of the previous layer
    var lastLayer = document.getElementById('layerAccordion').lastElementChild;
    var lastLayerOutputs = lastLayer.querySelector('#Outputs').value;
    layerClone.querySelector('#Inputs').value = lastLayerOutputs;

    // Append the cloned layer to the accordion
    document.getElementById('layerAccordion').appendChild(layerClone);
});

document.getElementById('exportButton').addEventListener('click', function () {
    // Create an object to store the configuration
    var configuration = {
        layers: [],
        optimizer: {
            name: document.getElementById('Optimiser').value, learningRate: document.getElementById('LearningRate').value,
            decay: document.getElementById('Decay').value,
            minLr: document.getElementById('MinLr').value,
        },

        loss: document.getElementById('Loss').value,
        metric: document.getElementById('Metrics').value
    };

    // Get all layers
    var layers = document.querySelectorAll('.layer');
    layers.forEach(function (layer) {
        var activation = layer.querySelector('#Activation1').value;
        var inputs = layer.querySelector('#Inputs').value;
        var outputs = layer.querySelector('#Outputs').value;

        // Add layer configuration to the array
        configuration.layers.push({
            activation: activation,
            inputs: inputs,
            outputs: outputs
        });
    });

    train(configuration, document.getElementById('trainData').files[0]);
});


function updateProgress(lossVal, metricVal, i, length) {
    document.getElementById('progressBar').style.width = Math.floor((i / length) * 100) + '%';
    document.getElementById('progressBar').setAttribute('aria-valuenow', Math.floor((i / length) * 100));
    document.getElementById('lossMetric').innerText = `Loss: ${JSON.stringify(lossVal)}, Metric: ${metricVal}`;
}

async function train(configuration, dataFile) {

    data = await parseData(dataFile);
    eval(convertToModelCode(configuration));
    trainer.onLoopDone = (lossVal, exp, out, loopNo, metricVal) => {
        (loopNo % 100 == 0) && console.log(JSON.stringify(lossVal.value), metricVal);

        (loopNo % 100 == 0) && updateProgress(lossVal.value, metricVal, loopNo, data.length)

        if (loopNo % 100 == 0) {
            var dataUri = "data:application/json;charset=utf-8;base64," + btoa(JSON.stringify([configuration, model.toJson()]));
            (document.getElementById('modelLink')).href = dataUri;
        }
    }


    const x = data.map((v) => [v.slice(1).map(i => Math.round(i / 255))]);
    const y = data.map((v) => [v[0]]);
    trainer.train(x, y, true)
    console.log(x);
}
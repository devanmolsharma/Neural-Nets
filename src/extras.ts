function convertToModelCode(config: any) {
    const modelCode = [];

    // Create model
    modelCode.push('var model = new Sequential();');

    // Add layers
    config.layers.forEach((layer: any) => {
        modelCode.push(`model.add(new Linear(${layer.inputs},${layer.outputs},'${layer.activation}'));`);
    });

    // Create optimizer
    const optimizerCode = `let opt = new SGD(model.getParameters(), {lr:${config.optimizer.learningRate || 1e-2}, decay:${config.optimizer.decay || 0.2}, minLr:${config.optimizer.minLr || 1e-4}});`;
    modelCode.push(optimizerCode);

    // Create loss
    const lossCode = `let lossFn = new ${config.loss}();`;
    modelCode.push(lossCode);

    // Create metric
    const metricCode = `const metric = new ${config.metric}(${config.metric === 'Accuracy' ? 100 : ''});`;
    modelCode.push(metricCode);

    // Create trainer
    modelCode.push('var trainer = new Trainer(model, opt, lossFn, metric);');

    return modelCode.join('\n');
}


async function loadModelFromJson(filePath: string) {

    // var model;

    const res = await fetch(filePath);
    const parsedJson = await res.json();
    const config = parsedJson[0];
    const weights = parsedJson[1];
    const modelCode = [];

    // Create model
    modelCode.push('var model = new Sequential();');

    // Add layers
    config.layers.forEach((layer: any) => {
        modelCode.push(`model.add(new Linear(${layer.inputs},${layer.outputs},'${layer.activation}'));`);
    });


    const code = modelCode.join('\n');

    eval(code);
    // @ts-ignore
    model.loadData(weights);

    // @ts-ignore
    return model;
}


async function loadModelData(net: Model, jsonPath: string) {
    net.loadData(await (await fetch(jsonPath)).text())
}

async function parseData(file: File) {
    const reader = new FileReader()
    reader.readAsText(file)
    let data: number[][] = [];
    return await new Promise<number[][]>((resolve, reject) => {
        reader.onloadend = () => {
            const lines = (reader.result as string).split('\n');
            lines.forEach((line) => {
                data.push([...line.split(',').map((n) => eval(n))])
    
            });
            return resolve(data);
        }
        
    })


}
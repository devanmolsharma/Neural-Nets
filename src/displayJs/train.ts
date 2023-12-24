/// <reference path="../TensorOperationsList.ts" />
/// <reference path="../Tensor.ts" />
/// <reference path="../Linear.ts" />
/// <reference path="../Sequential.ts" />
/// <reference path="../SGD.ts" />
/// <reference path="../Metric.ts" />
/// <reference path="../Trainer.ts" />




function createNeuralNetwork(): Sequential {
    const net = new Sequential();
    net.add(new Linear(784, 20, 'ReLU'));
    net.add(new Linear(20, 10, "Softmax", false));
    return net;
}

function createOptimizer(net: Sequential): SGD {
    return new SGD(net.getParameters(), { lr: 1e-2, decay: 0 });
}

async function train(data: number[][]) {
    const net = createNeuralNetwork();
    const optimiser = createOptimizer(net);
    const loss = new CrossEntropy();
    const metric = new Accuracy(100);

    const trainer = new Trainer(net, optimiser, loss, metric);


    trainer.onLoopDone = (loss, exp, out, loopNo, metric) => {
        (loopNo % 100 == 0) && console.log(JSON.stringify(loss.value), metric);
        if (metric > .90) {
            var dataUri = "data:application/json;charset=utf-8;base64," + btoa(net.toJson());
            (document.getElementById('model') as HTMLAnchorElement).href = dataUri;
        }
    }

    const x = data.map((v) => [v.slice(1)]);
    const y = data.map((v) => [v[0]]);
    trainer.train(x, y, true)
    console.log(x);

}

document.addEventListener('DOMContentLoaded', () => {
    let table = document.getElementById('mnistVisualiser');
    for (let i = 0; i < 28; i++) {
        let tr = document.createElement('tr') as HTMLTableRowElement;
        for (let j = 0; j < 28; j++) {
            tr.innerHTML += '<td></td>'
        }
        table.appendChild(tr);

    }

    document.getElementById('dataForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        let j = document.getElementById('data') as HTMLInputElement;
        const reader = new FileReader()
        reader.readAsText(j.files[0])
        let data: number[][] = [];
        reader.onloadend = () => {
            const lines = (reader.result as string).split('\n');
            lines.forEach((line) => {
                data.push([...line.split(',').map((n, i) => i == 0 ? parseInt(n) : Math.round(parseFloat(n) / 255))])

            });

            let table = document.getElementById('mnistVisualiser');
            for (let i = 1; i < data[0].length; i++) {
                const count = i - 1;
                let row = Math.floor(count / 28);
                let el = count - 28 * row;
                const value = data[0][i] * 255;
                (table.children[row].children[el] as HTMLTableCellElement).style.backgroundColor = `rgb(${value},${value},${value})`;

            }
            train(data);
        }
    })
})




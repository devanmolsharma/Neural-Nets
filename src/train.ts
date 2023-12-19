/// <reference path="TensorOperationsList.ts" />
/// <reference path="Tensor.ts" />
/// <reference path="Linear.ts" />
/// <reference path="Sequential.ts" />
/// <reference path="SGD.ts" />


async function train(data: number[][]) {
    const net = new Sequential();
    net.add(new Linear(784, 80, 'relu'));
    net.add(new Linear(80, 50, 'relu'));
    net.add(new Linear(50, 1));
    const optimiser = new SGD(net.getParameters(), { lr: 1e-4 });

    data.forEach((subData, i) => {
        setTimeout(((subData, i) => {
            return () => {
                const inp = new Tensor([[...subData.slice(1)]]);
                const out = net.forward([inp])
                const expected = new Tensor([[subData[0] * 255]]);
                let loss = new Subtract()([out, expected]);
                loss = new Multiply()([loss, loss]);
                loss = new Mean()([loss]);
                loss.backward();
                optimiser.step();
                optimiser.zero_grad();
                if (loss.value == Infinity) { throw 'Infinity Seen' }
                if (i % 500 == 0) {
                    console.log(loss.value, expected.value[0] as number[][0], out.value[0] as number[][0]);
                    var dataUri = "data:application/json;charset=utf-8;base64," + btoa(net.toJson());
                    (document.getElementById('model') as HTMLAnchorElement).href = dataUri;
                }
            }

        })(subData, i), 0);
        // throw ' ';

    })

}
// let s = new Sequential();

// s.add(new Linear(1, 4,'relu'))
// s.add(new Linear(4, 1,'relu'))
// s.add(new Linear(1, 1,'relu'))
// const optimiser = new SGD(s.getParameters(), {lr:1e-4});

// for (let i = 0; i < 1000; i++) {
//     const out = s.forward([new Tensor([[2]])])
//     const expected = new Tensor([[4]]);
//     let loss = new Subtract()([out, expected]);
//     loss = new Mean()([loss])
//     loss.backward();
//     optimiser.step();
//     console.log(loss.value);

// }

// console.log(s.toJson());

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
                data.push([...line.split(',').map(n => parseFloat(n) / 255)])

            });

            let table = document.getElementById('mnistVisualiser');
            for (let i = 1; i < data[0].length; i++) {
                const count = i - 1;
                let row = Math.floor(count / 28);
                let el = count - 28 * row;
                const value = data[0][i] * 255;
                (table.children[row].children[el] as HTMLTableCellElement).style.backgroundColor = `rgb(${value},${value},${value})`;

            }
            setTimeout(() => {
                train(data);

            }, 1000);

        }
    })
})




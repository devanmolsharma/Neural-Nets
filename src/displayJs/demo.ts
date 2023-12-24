let draw = false;
let arr = TensorUtils.filledArray([28, 28], 0) as number[][]

let model = createNeuralNetwork();
function argmax(arr: number[]) {
    return (arr).findIndex((v) => v == Math.max(...arr))
}

async function loadData() {

    model.loadData(await (await fetch('http://localhost/model.php')).text())


}

document.addEventListener('DOMContentLoaded', () => {

    let table = document.getElementById('drawBase');
    if (table) {
        for (let i = 0; i < 28; i++) {
            let tr = document.createElement('tr') as HTMLTableRowElement;
            for (let j = 0; j < 28; j++) {
                tr.innerHTML += `<td class="pixel ${i}-${j}"></td>`
            }
            table.appendChild(tr);

        }

        const pixels = document.getElementsByClassName('pixel');
        for (const pixel of pixels) {
            pixel.addEventListener('pointermove', (e) => {
                if (draw) {
                    if (!(e.target as HTMLTableCellElement).classList.contains('black')) {
                        const [x, y] = (e.target as HTMLTableCellElement).classList[1].split('-').map(i => parseInt(i));
                        (e.target as HTMLTableCellElement).classList.add('black');
                        (arr as any)[x][y] = 1;

                        let display = document.getElementById('mnistVisualiser');
                        for (let i = 1; i < 28; i++) {
                            for (let j = 0; j < 28; j++) {
                                const value = arr[i][j] * 255;
                                (display.children[i].children[j] as HTMLTableCellElement).style.backgroundColor = `rgb(${value},${value},${value})`;

                            }

                        }

                        const pred = model.forward([new Tensor([arr.flat()])]).value.flat() as any as number[];
                        const highest = argmax(pred);
                        console.log(model.layers[1]);

                        document.getElementById("prediction").innerHTML = `
    <h2 class="mb-4">Prediction</h2>
    <div class="alert alert-info" role="alert">
        It seems to be <strong>${highest}</strong>
    </div>
    <div class="mt-4">
        ${Object.entries(pred).map(([digit, probability]) => `
            <div class="progress mb-3">
                <div class="progress-bar" role="progressbar" style="width: ${probability * 100}%;"
                    aria-valuenow="${probability * 100}" aria-valuemin="0" aria-valuemax="100">
                    ${digit}
                </div>
            </div>
        `).join('')}
    </div>
`;



                        // 'I guess it is '+ argmax(model.forward([new Tensor([arr.flat()])]).value.flat() as any as number[]);

                    }
                }

            })

            pixel.addEventListener('pointerdown', (e) => {
                draw = true;
            })

            window.addEventListener('pointerup', (e) => {
                draw = false;
            })
        }

        loadData();
    }

})

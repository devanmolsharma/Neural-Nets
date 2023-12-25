let draw = false;
let arr = TensorUtils.filledArray([28, 28], 0)
var model;
async function load() {
    model = await loadModelFromJson('./mnist.json');

    console.log('====================================');
    console.log(model);
    console.log('====================================');
}

let brushSize = 1;


document.addEventListener('DOMContentLoaded', () => {

    load()

    let table1 = document.getElementById('mnistVisualiser');
    for (let i = 0; i < 28; i++) {
        let tr = document.createElement('tr');
        for (let j = 0; j < 28; j++) {
            tr.innerHTML += '<td></td>'
        }
        table1.appendChild(tr);

    }

    let table = document.getElementById('drawBase');
    if (table) {
        for (let i = 0; i < 28; i++) {
            let tr = document.createElement('tr');
            for (let j = 0; j < 28; j++) {
                tr.innerHTML += `<td class="pixel ${i}-${j}"></td>`
            }
            table.appendChild(tr);

        }

        const pixels = document.getElementsByClassName('pixel');

        document.getElementById('clear').addEventListener('click',() => {
            arr = TensorUtils.filledArray([28, 28], 0);
            for (const pixel of pixels) {
                pixel.classList.contains('black') && pixel.classList.remove('black');
            }
        });

        for (const pixel of pixels) {
            pixel.addEventListener('pointermove', (e) => {
                if (draw) {
                    // Use the brushSize to update adjacent pixels
                    const [x, y] = (e.target).classList[1].split('-').map(i => parseInt(i));
                    for (let i = x - brushSize + 1; i <= x + brushSize - 1; i++) {
                        for (let j = y - brushSize + 1; j <= y + brushSize - 1; j++) {
                            if (i >= 0 && i < 28 && j >= 0 && j < 28) {
                                const targetPixel = document.getElementsByClassName(`${i}-${j}`)[0];
                                if (!targetPixel.classList.contains('black')) {
                                    targetPixel.classList.add('black');
                                    arr[i][j] = 1;
                                }
                            }
                        }

                        // Update the display
                        let display = document.getElementById('mnistVisualiser');
                        for (let i = 1; i < 28; i++) {
                            for (let j = 0; j < 28; j++) {
                                const value = arr[i][j] * 255;
                                (display.children[i].children[j]).style.backgroundColor = `rgb(${value},${value},${value})`;
                            }
                        }

                        const pred = model.forward([new Tensor([arr.flat()])]).value.flat();
                        const highest = TensorUtils.argmax(pred);
                        console.log(model.layers[1]);
                        console.log('====================================');
                        console.log(pred);
                        console.log('====================================');

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
    }

})

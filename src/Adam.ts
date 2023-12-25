/// <reference path="Optimiser.ts" />

class Adam extends Optimiser {
    private lr: number;
    private minLr: number;
    private momentum: number;

    constructor(_parameters: Map<String, Tensor>[], { ...kwargs }) {
        super(_parameters, kwargs);
    };
    processGradient(gradient: Tensor, step: number, kwargs: any): Tensor {
        throw "Not Implemented Yet"
    }

    step(): void {
        super.step();
    }
}
abstract class Metric {
    abstract forward(actual: Tensor, pred: Tensor, loop: number): number;
}

class Accuracy extends Metric {
    private corrects: number = 0;

    constructor(private afterIterations: number = 100) {
        super();
    }

    forward(actual: Tensor, pred: Tensor, loop: number): number {
        const accuracy = this.corrects / 100;
        this.reset(loop)
        if(TensorUtils.argmax(actual.value.flat(20) as number[]) == TensorUtils.argmax(pred.value.flat(20) as number[])){
            this.corrects++;
        }
        return accuracy;
    }

    reset(loop: number) {
        if (loop % this.afterIterations == 0) {
            this.corrects = 0;
        }
    }

}
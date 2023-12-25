class Trainer {
    constructor(private net: Model, private optimiser: Optimiser, private loss: TensorOperation, private metric: Metric | null = null) { }
    onTrainingDone(net: Model): void { }
    onLoopDone(loss: Tensor, expected: Tensor, out: Tensor, loopNo: number, metric: number): void { }

    loop(x: NumArray, y: NumArray, loopNo: number) {
        let out = this.net.forward([new Tensor(x)]);

        const expected = new Tensor(y);

        let loss = this.loss([out, expected]);

        loss = new Mean()([loss]);

        loss.backward();
        this.optimiser.step();
        this.optimiser.zero_grad();

        if (loss.value == Infinity) {
            throw 'Infinity Seen';
        }


        this.onLoopDone(loss, expected, out, loopNo, this.calculateMetric(expected, out, loopNo))
    }

    calculateMetric(expected: Tensor, out: Tensor, loopNo: number) {
        return this.metric.forward(expected, out, loopNo);
    }

    train(x: NumArray, y: NumArray, oneHotEncode: boolean = false) {

        for (let i = 0; i < x.length; i++) {
            const inp = x[i];
            let exp = y[i];
            if (oneHotEncode) {
                let oneHot = new Array(10).fill(0);

                oneHot[exp as number] = 0.8;
                exp = [oneHot];
            }

            setTimeout(((inp: NumArray, exp: NumArray, i: number): any => {
                return () => this.loop(inp as NumArray, exp as NumArray, i)
            })(inp as NumArray, exp as NumArray, i), 0)

        }

        this.onTrainingDone(this.net);

    }
}
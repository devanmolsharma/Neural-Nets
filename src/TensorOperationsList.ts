class Sum extends TensorOperation {
    private tensorCount: number;

    private addArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.addArrays(v, b[i]) : v + b[i]);
    }

    private sum(arrays: any) {
        return arrays.reduce((a: any, b: any) => this.addArrays(a, b));
    }

    public forward(tensors: NumArray[]): NumArray {
        const sum = this.sum(tensors);
        return sum;

    }

    public backward(gradient: NumArray): NumArray[] {
        return Array(this.tensorCount).fill(gradient);
    }

    public setup(tensors: Tensor[]): void {
        this.tensorCount = tensors.length;
    }
}


class Mean extends TensorOperation {
    private shape: number[];
    private elementCount: number;

    public forward(tensors: NumArray[]): NumArray {
        const sum = (tensors.flat().reduce((i, j) => (i as number) + (j as number)) as number) / (this.elementCount)
        return [sum];

    }

    public backward(gradient: NumArray): NumArray[] {
        return [TensorFactory.filled(this.shape, gradient[0] as number).value];
    }

    public setup(tensors: Tensor[]): void {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y)
    }
}
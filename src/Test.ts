class Sum extends TensorOperation {

    private addArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.addArrays(v, b[i]) : v + b[i]);
    }

    private sum(arrays: any) {
        return arrays.reduce((a: any, b: any) => this.addArrays(a, b));
    }

    public forward(...tensors: NumArray[]): NumArray {
        const sum = this.sum(tensors);
        return sum;

    }

    public backward(gradient: NumArray): NumArray[] {
        return [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
    }
}

class Mean extends TensorOperation {

    public forward(...tensors: NumArray[]): NumArray {
        const sum = tensors.flat().reduce((i, j) => (i as number) + (j as number))
        return [sum];

    }

    public backward(gradient: NumArray): NumArray[] {
        return [[1]];
    }
}

const [t1, t2, t3] = [new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4])];
let j = new Sum()(t1, t2, t3)
j = new Mean()(j)
j.backward()
console.log(j.gradientHandler.gradient.toString());
// Class representing the sum operation for tensors
class Sum extends TensorOperation {
    // Number of tensors involved in the sum operation
    private tensorCount: number;

    // Helper function to add arrays element-wise
    private addArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.addArrays(v, b[i]) : v + b[i]);
    }

    // Function to calculate the sum of arrays
    private sum(arrays: any) {
        return arrays.reduce((a: any, b: any) => this.addArrays(a, b));
    }

    // Forward pass of the sum operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the sum of the tensors
        const sum = this.sum(tensors);
        return sum;
    }

    // Backward pass of the sum operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient to all input tensors
        return Array(this.tensorCount).fill(gradient);
    }

    // Setup method to initialize the tensor count
    public setup(tensors: Tensor[]): void {
        this.tensorCount = tensors.length;
    }
}

// Class representing the mean operation for tensors
class Mean extends TensorOperation {
    // Shape of the tensors involved in the mean operation
    private shape: number[];

    // Total number of elements in the tensors
    private elementCount: number;

    // Forward pass of the mean operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the mean of the tensors
        const sum = (tensors[0].flat().reduce((i, j) => (i as number) + (j as number)) as number) / (this.elementCount);

        return [sum];
    }

    // Backward pass of the mean operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient equally to all elements in the tensors
        return [TensorFactory.filled(this.shape, gradient[0] as number).value];
    }

    // Setup method to initialize shape and element count
    public setup(tensors: Tensor[]): void {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y);
    }
}


// Class representing the mean operation for tensors
class Matmul extends TensorOperation {
    // Shape of the tensors involved in the Matmul operation
    private shape1: number[];
    private shape2: number[];
    private t1: number[][];
    private t2: number[][];

    static tensorMul(tensors: Array<number[][]>) {
        // Calculate the mean of the tensors
        let finalArr = TensorFactory.filledArray([tensors[0].length, tensors[1][0].length], 0) as number[][];

        for (let row = 0; row < finalArr.length; row++) {
            for (let col = 0; col < finalArr[0].length; col++) {
                let sum: number = 0;
                for (let i = 0; i < tensors[0][row].length; i++) {
                    sum += tensors[0][row][i] * tensors[1][i][col]

                }
                finalArr[row][col] = sum;
            }
        }
        return finalArr;
    }

    // Forward pass of the mean operation
    public forward(tensors: NumArray[]): NumArray {

        this.t1 = tensors[0] as number[][];
        this.t2 = tensors[1] as number[][];

        return Matmul.tensorMul([this.t1, this.t2]);


    }

    public backward(gradient: NumArray): NumArray[] {
        let grads = [Matmul.tensorMul([gradient, TensorFactory.transpose(this.t2)]), Matmul.tensorMul([this.t1, gradient as number[][]])];

        return grads;
    }

    // Do Basic checks and store rensor shapes
    public setup(tensors: Tensor[]): void {
        if (tensors.length != 2) {
            throw "this Operation requires exactly two Tensors";
        }

        this.shape1 = tensors[0].shape;
        this.shape2 = tensors[1].shape;

        if (this.shape1[1] != this.shape2[0]) {
            throw `Cannot multiply matrices with shapes ${this.shape1} and ${this.shape2}`;
        }
    }
}

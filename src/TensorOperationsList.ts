/// <reference path="TensorOperation.ts" />
// Class representing the sum operation for tensors
class Add extends TensorOperation {
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
        const sum = (tensors[0].flat(20).reduce((i, j) => (i as number) + (j as number)) as number) / (this.elementCount);

        return [sum];
    }

    // Backward pass of the mean operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient equally to all elements in the tensors
        return [TensorUtils.filledArray(this.shape, gradient[0] as number) as NumArray];
    }

    // Setup method to initialize shape and element count
    public setup(tensors: Tensor[]): void {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y);
    }
}

class Transpose extends TensorOperation {
    // Shape of the tensors involved in the mean operation
    private shape: number[];

    // Total number of elements in the tensors
    private elementCount: number;

    // Forward pass of the Transpose operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the Transpose of the tensors
        const transpose = TensorUtils.transpose(tensors[0]);

        return transpose;
    }

    // Backward pass of the Transpose operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient equally to all elements in the tensors
        const transpose = TensorUtils.transpose(gradient);

        return [transpose];
    }

    // Setup method to initialize shape and element count
    public setup(tensors: Tensor[]): void {
    }
}

// Class representing the Matrix Multiplication operation for tensors
class Matmul extends TensorOperation {
    // Shape of the tensors involved in the Matmul operation
    private t1: number[][];
    private t2: number[][];

    // Static method to perform matrix multiplication of two tensors
    static tensorMul(tensors: Array<number[][]>) {
        // Calculate the mean of the tensors
        let finalArr = TensorUtils.filledArray([tensors[0].length, tensors[1][0].length], 0) as number[][];

        for (let row = 0; row < finalArr.length; row++) {
            for (let col = 0; col < finalArr[0].length; col++) {
                let sum: number = 0;
                for (let i = 0; i < tensors[0][row].length; i++) {
                    sum += tensors[0][row][i] * tensors[1][i][col];
                }
                finalArr[row][col] = sum;
            }
        }
        return finalArr;
    }

    // Forward pass of the Matmul operation
    public forward(tensors: NumArray[]): NumArray {
        this.t1 = tensors[0] as number[][];
        this.t2 = tensors[1] as number[][];
        return Matmul.tensorMul([this.t1, this.t2]);
    }

    // Backward pass of the Matmul operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [Matmul.tensorMul([gradient, TensorUtils.transpose(this.t2)]), Matmul.tensorMul([TensorUtils.transpose(this.t1), gradient as number[][]])];
        return grads;
    }

    // Do basic checks and store tensor shapes
    public setup(tensors: Tensor[]): void {
        if (tensors.length != 2) {
            throw "This operation requires exactly two tensors.";
        }

        const shape1 = tensors[0].shape;
        const shape2 = tensors[1].shape;

        if (shape1[1] != shape2[0]) {
            throw `Cannot multiply matrices with shapes ${shape1} and ${shape2}`;
        }
    }
}

// Class representing the Min operation for tensors
class Min extends TensorOperation {
    private limit: number;
    private tensor: NumArray;

    // Static method to find the minimum value between tensor and limit
    static min(tensor: any, limit: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.min(val, limit));
        }
        return tensor as any as number < limit ? tensor : limit;
    }
    static calcGrad(tensor: any, gradient: any, limit: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any, i: number) => this.calcGrad(val, gradient[i], limit));
        }
        return tensor as any as number < limit ? gradient : limit;
    }

    // Forward pass of the Min operation
    public forward(tensors: NumArray[]): NumArray {
        this.limit = tensors[1][0] as number;
        this.tensor = tensors[0];
        return Min.min(tensors[0], tensors[1][0] as number) as NumArray;
    }

    // Backward pass of the Min operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [Min.calcGrad(this.tensor, gradient, this.limit) as NumArray, [0]];
        return grads;
    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
        if (tensors.length != 2) {
            throw "This operation requires exactly two tensors.";
        }

        if (JSON.stringify(tensors[1].shape) != JSON.stringify([1])) {
            throw "Tensor 2 should have shape [1]";
        }
    }
}

// Max Operation for relu activation later
class Max extends TensorOperation {
    private _limit: number;
    public get limit(): number {
        return this._limit;
    }
    public set limit(value: number) {
        this._limit = value;
    }
    private tensor: NumArray;

    // Static method to find the maximum value between tensor and limit
    static max(tensor: any, limit: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.max(val, limit));
        }
        return tensor as any as number > limit ? tensor : limit;
    }

    // Static method to calculate the gradient for the Max operation
    static calcGrad(tensor: any, gradient: any, limit: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any, i: number) => this.calcGrad(val, gradient[i], limit));
        }
        return tensor as any as number > limit ? gradient : 0;
    }

    // Forward pass of the Max operation
    public forward(tensors: NumArray[]): NumArray {
        this.tensor = tensors[0];
        return Max.max(tensors[0], this.limit) as NumArray;
    }

    // Backward pass of the Max operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [Max.calcGrad(this.tensor, gradient, this.limit) as NumArray, [0]];
        return grads;
    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
        this.limit = tensors[1] ? tensors[1].value[0] as number : 0;
        if (tensors[1] && (JSON.stringify(tensors[1].shape) != JSON.stringify([1]))) {
            throw "Tensor 2 should have shape [1]";
        }
    }
}

// Max Operation for relu activation later
class Normalize extends TensorOperation {
    private max: number;
    mean: number;
    variance: any;
    stdDev: number;

    // Static method to find the maximum value between tensor and limit
    static norm(tensor: any, max: number, mean: number, variance: number, stdiv: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.norm(val, max, mean, variance, stdiv));
        }
        return (tensor as any as number - mean) / stdiv;
    }


    // Forward pass of the Max operation
    public forward(tensors: NumArray[]): NumArray {
        const flatt0 = tensors[0].flat(20);
        this.max = Math.max(flatt0.reduce((x, y) => (x > y) ? x : y) as number, 1);
        this.mean = flatt0.reduce((x: number, y: number) => x + y) as number / flatt0.length;
        this.variance = flatt0.reduce((sum: number, val: number) => sum + Math.pow(val - this.mean, 2), 0) as number / flatt0.length;
        this.stdDev = Math.sqrt(this.variance);

        return Normalize.norm(tensors[0], this.max, this.mean, this.variance, this.stdDev) as NumArray;
    }

    // Static method to calculate the gradient for the Max operation
    static calcGrad(gradient: any, max: number, mean: number, variance: number, stdiv: number): NumArray | number {
        if (Array.isArray(gradient as any)) {
            return gradient.map((val: any, i: number) => this.calcGrad(val, max, mean, variance, stdiv));
        }
        return (gradient as any as number / stdiv);
    }

    // Backward pass of the Max operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [Normalize.calcGrad(gradient, this.max, this.mean, this.variance, this.stdDev) as NumArray];
        return grads;
    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
    }
}

class Rescale extends TensorOperation {
    private max: number;

    // Static method to find the maximum value between tensor and limit
    static rescale(tensor: any, max: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.rescale(val, max));
        }
        return (tensor as any as number) / max;
    }


    // Forward pass of the Max operation
    public forward(tensors: NumArray[]): NumArray {
        const flatt0 = tensors[0].flat(20);
        this.max = Math.max(flatt0.reduce((x, y) => (x > y) ? x : y) as number, 1);

        return Rescale.rescale(tensors[0], this.max) as NumArray;
    }

    // Static method to calculate the gradient for the Max operation
    static calcGrad(gradient: any, max: number): NumArray | number {
        if (Array.isArray(gradient as any)) {
            return gradient.map((val: any, i: number) => this.calcGrad(val, max));
        }
        
        return (gradient as any as number / max);
    }

    // Backward pass of the Max operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [Rescale.calcGrad(gradient, this.max) as NumArray];
        return grads;
    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
    }
}

class NanToNum extends TensorOperation {
    private limit: number;
    private tensor: NumArray;

    // Static method to find the maximum value between tensor and limit
    static unNanify(tensor: any): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.unNanify(val));
        }
        if (isNaN(tensor) || tensor == null) return 0;
        return tensor;
    }

    // Forward pass of the Max operation
    public forward(tensors: NumArray[]): NumArray {
        this.tensor = tensors[0];
        return NanToNum.unNanify(tensors[0]) as NumArray;
    }

    // Backward pass of the Max operation
    public backward(gradient: NumArray): NumArray[] {
        let grads = [NanToNum.calcGrad(this.tensor, gradient) as NumArray, [0]];
        return grads;
    }

    static calcGrad(tensor: any, gradient: any): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any, i: number) => this.calcGrad(val, gradient[i]));
        }
        if (isNaN(tensor)) return 0;
        return 1;
    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
    }
}

// Class representing the Min operation for tensors
class Sigmoid extends TensorOperation {
    private result: NumArray;
    private shape: number[];

    // Static method to find the minimum value between tensor and limit
    static sig(tensor: any): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.sig(val));
        }
        return 1 / (1 + Math.exp(-(tensor as any as number)));
    }

    // Forward pass of the Min operation
    public forward(tensors: NumArray[]): NumArray {
        this.result = Sigmoid.sig(tensors[0]) as NumArray;
        return structuredClone(this.result);
    }

    // Backward pass of the Min operation
    public backward(gradient: NumArray): NumArray[] {
        const res = [Multiply.product([
            gradient,
            Multiply.product([
                structuredClone(this.result), Subtract.diff([TensorUtils.filledArray(this.shape, 1), structuredClone(this.result)]
                )
            ])
        ])
        ];
        return res;

    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {
        this.shape = tensors[0].shape;
    }
}

class Softmax extends TensorOperation {
    private result: NumArray;
    private shape: number[];
    private sum: number;

    // Static method to find the minimum value between tensor and limit
    static softmax(tensor: any, sum: number): NumArray | number {
        if (Array.isArray(tensor as any)) {
            return tensor.map((val: any) => this.softmax(val, sum));
        }

        return Math.exp(tensor as any as number) / sum;
    }

    // Forward pass of the Min operation
    public forward(tensors: NumArray[]): NumArray {
        this.sum = tensors[0].flat(20).reduce((prev, curr) => {
            return (prev as number) + Math.exp((curr as number))
        }) as number;


        this.result = Softmax.softmax(tensors[0], this.sum) as NumArray;
        return structuredClone(this.result);
    }

    // Backward pass of the Min operation
    public backward(gradient: NumArray): NumArray[] {

        let grad = TensorUtils.filledArray([this.shape[1], this.shape[1]], 0) as number[][];

        for (let x = 0; x < this.shape[1]; x++) {
            for (let y = 0; y < this.shape[1]; y++) {
                if (x == y) {
                    const val = this.result.flat(20)[x] as number
                    grad[x][y] = val * (1 - val);
                } else {
                    const val1 = this.result.flat(20)[x] as number
                    const val2 = this.result.flat(20)[y] as number
                    grad[x][y] = -1 * val1 * val2;
                }
            }

        }

        return [Matmul.tensorMul([gradient as number[][], grad])];

    }

    // Do basic checks
    public setup(tensors: Tensor[]): void {

        this.shape = tensors[0].shape;

        tensors[0] = new Rescale()([tensors[0]]);
    }
}

class ReLU extends Max {

}

class LeakyReLU extends Max {
    public setup(tensors: Tensor[]): void {
        super.limit = -0.01;
    }
}

// Class representing the Subtraction operation for tensors
class Subtract extends TensorOperation {
    // Helper function to subtract arrays element-wise
    public static subtract2DArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.subtract2DArrays(v, b[i]) : v - b[i]);
    }

    // Function to calculate the difference of arrays
    public static diff(arrays: any) {
        return arrays.reduce((a: any, b: any) => Subtract.subtract2DArrays(a, b));
    }

    // Forward pass of the Subtract operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the difference of the tensors
        const diff = Subtract.diff(tensors);
        return diff;
    }

    // Backward pass of the Subtract operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient to all input tensors
        return [gradient, TensorUtils.reshape((gradient.flat(20) as number[]).map((i) => -i) as NumArray, TensorUtils.calculateShape(gradient) as number[]) as NumArray];
    }

    // Setup method to initialize the tensor count
    public setup(tensors: Tensor[]): void {
        if (tensors.length != 2) {
            throw "Subtraction is only valid operation for two tensors.";
        }
    }
}

class CrossEntropy extends TensorOperation {
    private tensors: NumArray[];
    // Helper function to subtract arrays element-wise
    public static crossEntropy2DArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.crossEntropy2DArrays(v, b[i]) : (-Math.log(1 + 1e-8 - (Math.min(v,1) - b[i]) ** 2)));
    }


    public static crossEntropyBack(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.crossEntropyBack(v, b[i]) : ((2 * (v - b[i])) / (1 + 1e-8 - ((v - b[i]) ** 2))));
    }

    // Function to calculate the difference of arrays
    public static ce(arrays: any) {
        return arrays.reduce((a: any, b: any) => CrossEntropy.crossEntropy2DArrays(a, b));
    }

    // Forward pass of the Subtract operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the difference of the tensors
        this.tensors = tensors;
        return CrossEntropy.ce(tensors);
    }

    // Backward pass of the Subtract operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient to all input tensors
        return [CrossEntropy.crossEntropyBack(this.tensors[0], this.tensors[1]), gradient];
    }

    // Setup method to initialize the tensor count
    public setup(tensors: Tensor[]): void {
        if (tensors.length != 2) {
            throw "Subtraction is only valid operation for two tensors.";
        }
    }
}

// Class representing the Multiply operation for tensors
class Multiply extends TensorOperation {
    private tensors: NumArray[];

    // Helper function to multiply arrays element-wise
    static multiplyArrays(a: any, b: any) {
        return a.map((v: any, i: any) => Array.isArray(v) ? this.multiplyArrays(v, b[i]) : v * b[i]);
    }

    // Function to calculate the product of arrays
    static product(arrays: any) {
        return arrays.reduce((a: any, b: any) => this.multiplyArrays(a, b));
    }

    // Forward pass of the Multiply operation
    public forward(tensors: NumArray[]): NumArray {
        // Calculate the product of the tensors
        const res = Multiply.product(tensors);
        return res;
    }

    // Backward pass of the Multiply operation
    public backward(gradient: NumArray): NumArray[] {
        // Distribute the gradient to all input tensors
        if (this.tensors.length == 2) {
            return [Multiply.product([this.tensors[1], gradient]), Multiply.product([this.tensors[0], gradient])];
        } else {
            let grads = [];
            for (const array of this.tensors) {
                grads.push(Multiply.product([...this.tensors.filter((v) => v != array), gradient]))
            }
            return grads
        }
    }

    // Setup method to initialize the tensors
    public setup(tensors: Tensor[]): void {
        if (tensors.length < 2) {
            throw "Multiplication is only valid for over two tensors.";
        };

        this.tensors = tensors.map((e) => e.value);
    }
}

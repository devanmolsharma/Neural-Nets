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
        const sum = (tensors.flat().reduce((i, j) => (i as number) + (j as number)) as number) / (this.elementCount);
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

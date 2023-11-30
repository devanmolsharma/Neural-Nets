// Factory class for creating tensors with different initial values and shapes
class TensorFactory {
    // Create a filled array with the specified shape and fill value
    static filledArray(shape: number[], fillValue = 0): NumArray | number {
        // If the shape is empty, return the fill value
        if (shape.length === 0) {
            return fillValue;
        }

        // Create a new array with the specified shape
        let array = new Array(shape[0]);

        // Recursively fill the array with nested arrays
        for (let i = 0; i < array.length; i++) {
            array[i] = this.filledArray(shape.slice(1), fillValue);
        }

        return array;
    }

    // Create a tensor with all elements set to 1 and the specified shape
    static ones(shape: number[]): Tensor {
        return new Tensor(this.filledArray(shape, 1) as NumArray);
    }

    // Create a tensor with all elements set to 0 and the specified shape
    static zeros(shape: number[]): Tensor {
        return new Tensor(this.filledArray(shape, 0) as NumArray);
    }

    // Create a tensor with all elements set to the specified fill value and shape
    static filled(shape: number[], fillValue: number): Tensor {
        return new Tensor(this.filledArray(shape, fillValue) as NumArray);
    }
}

// Factory class for creating tensors with different initial values and shapes
class TensorUtils {
    // Reshape a given array into the specified shape
    static reshape(array: NumArray, shape: number[]): NumArray | number {
        let result = this.filledArray(shape);
        let flat = this.flatten(array);
        for (let i = 0; i < flat.length; i++) {
            let indices = this.getIndices(i, shape);
            this.setElement(result, indices, flat[i]);
        }
        return result;
    }

    // Transpose a given 2D array
    static transpose(array: any): any {
        let result = this.filledArray([array[0].length, array.length]) as any;
        for (let i = 0; i < array.length; i++) {
            for (let j = 0; j < array[i].length; j++) {
                result[j][i] = array[i][j];
            }
        }
        return result;
    }

    // Helper method to flatten an array
    private static flatten(array: any): any[] {
        return array.reduce((acc: any, val: any) => Array.isArray(val) ? acc.concat(this.flatten(val)) : acc.concat(val), []);
    }

    // Helper method to get multi-dimensional indices from a flat index
    private static getIndices(index: number, shape: number[]): number[] {
        let indices = [];
        for (let i = shape.length - 1; i >= 0; i--) {
            indices.unshift(index % shape[i]);
            index = Math.floor(index / shape[i]);
        }
        return indices;
    }

    // Helper method to set an element at the specified multi-dimensional indices
    private static setElement(array: any, indices: number[], value: any) {
        for (let i = 0; i < indices.length - 1; i++) {
            array = array[indices[i]];
        }
        array[indices[indices.length - 1]] = value;
    }



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

    static calculateShape(array: any, shape: number[] = []): number[] {
        shape.push(array.length);
        if (Array.isArray(array[0])) {
            TensorUtils.calculateShape(array[0], shape);
        }
        return shape as number[];
    }
}

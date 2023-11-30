class TensorFactory {
    static filledArray(shape: number[], fillValue = 0): NumArray | number {

        if (shape.length === 0) {
            return fillValue;
        }

        let array = new Array(shape[0]);

        for (let i = 0; i < array.length; i++) {
            array[i] = this.filledArray(shape.slice(1), fillValue);
        }
        return array;
    }

    static ones(shape: number[]): Tensor {
        return new Tensor(this.filledArray(shape, 1) as NumArray);
    }

    static zeros(shape: number[]): Tensor {
        return new Tensor(this.filledArray(shape, 0) as NumArray);
    }

    static filled(shape: number[], fillValue: number): Tensor {
        return new Tensor(this.filledArray(shape, fillValue) as NumArray);
    }

}
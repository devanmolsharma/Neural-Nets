type NestedArray<T> = T | NestedArray<T>[];

class Tensor {
    private _gradientHandler:GradientHandler = new GradientHandler(this);

    constructor(private _value: NestedArray<number>) {
    }


    public get value(): NestedArray<number> {
        return structuredClone(this._value);
    }


    public set value(value: NestedArray<number>) {
        this._value = value;
    }


}
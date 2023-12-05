declare class GradientHandler {
    private _tensor;
    private Operation;
    private _children;
    private _gradient;
    constructor(_tensor: Tensor);
    get children(): Tensor[];
    get gradient(): Tensor;
    set gradient(value: Tensor);
    registerChildren(children: Tensor[]): void;
    registerOperation(operation: TensorOperation): void;
    backward(previousGradient?: Tensor): void;
}
type NestedArray<T> = T | Array<NestedArray<T>>;
interface NumArray extends Array<NestedArray<number>> {
}
declare class Tensor {
    private _value;
    gradientHandler: GradientHandler;
    constructor(_value: NumArray);
    get value(): NumArray;
    set value(value: NumArray);
    get shape(): number[];
    clone(): Tensor;
    private _calculateShape;
    backward(gradient: Tensor): void;
    toString(): string;
}
declare class TensorFactory {
    static reshape(array: NumArray, shape: number[]): NumArray | number;
    static transpose(array: any): any;
    private static flatten;
    private static getIndices;
    private static setElement;
    static filledArray(shape: number[], fillValue?: number): NumArray | number;
    static ones(shape: number[]): Tensor;
    static zeros(shape: number[]): Tensor;
    static filled(shape: number[], fillValue: number): Tensor;
}
declare abstract class TensorOperation extends Function {
    private _bound;
    constructor();
    verify(tensors: Tensor[]): void;
    abstract forward(tensors: NumArray[], ...kwargs: any): NumArray;
    abstract backward(gradient: NumArray): NumArray[];
    abstract setup(tensors: Tensor[], ...kwargs: any): void;
    private _call;
    getGradient(gradient: Tensor): Tensor[];
}
declare class Sum extends TensorOperation {
    private tensorCount;
    private addArrays;
    private sum;
    forward(tensors: NumArray[]): NumArray;
    backward(gradient: NumArray): NumArray[];
    setup(tensors: Tensor[]): void;
}
declare class Mean extends TensorOperation {
    private shape;
    private elementCount;
    forward(tensors: NumArray[]): NumArray;
    backward(gradient: NumArray): NumArray[];
    setup(tensors: Tensor[]): void;
}
declare class Matmul extends TensorOperation {
    private shape1;
    private shape2;
    private t1;
    private t2;
    static tensorMul(tensors: Array<number[][]>): number[][];
    forward(tensors: NumArray[]): NumArray;
    backward(gradient: NumArray): NumArray[];
    setup(tensors: Tensor[]): void;
}
declare const t1: Tensor, t2: Tensor;
declare let j: any;
declare let mean: any;

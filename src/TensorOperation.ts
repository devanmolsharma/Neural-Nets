
abstract class TensorOperation extends Function {
    private _bound: TensorOperation;

    constructor() {
        super('...args', 'return this._bound._call(...args)')

        this._bound = this.bind(this)

        return this._bound
    }


    public verify(tensors: Tensor[]): void {
        const out = this._call(tensors);
        const grad = this.getGradient(out);
        if (grad.length != tensors.length) {
            throw "backward function does not return gradients of all tensors";
        }
        for (const i in tensors) {
            if (JSON.stringify(tensors[i].shape) != JSON.stringify(grad[i].shape)) {
                throw `gradient shape mismatch on element at index ${i}.\n required shape: ${tensors[i].shape} but got shape ${grad[i].shape}`;
            }
        }
        console.log('Shapes verified for Class:' + this.constructor.name);

    }


    public abstract forward(tensors: NumArray[], ...kwargs: any): NumArray;


    public abstract backward(gradient: NumArray): NumArray[];

    public abstract setup(tensors: Tensor[], ...kwargs: any): void;

    private _call(tensors: Tensor[], ...kwargs: any): Tensor {
        this.setup(tensors, ...kwargs);
        const out = new Tensor(this.forward((tensors.map((t) => t.value)), ...kwargs));
        out.gradientHandler.registerChildren(tensors);
        out.gradientHandler.registerOperation(this);
        return out;
    }

    public getGradient(gradient: Tensor): Tensor[] {
        const grads = this.backward(gradient.value);
        return grads.map((e) => new Tensor(e));
    }

}

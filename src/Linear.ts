/// <reference path="Layer.ts" />
// a simple impelentation of Linear Layer

class Linear extends Layer {
    private weights: Tensor;
    private biases: Tensor;
    private transposer: Transpose;
    private adder: Add;
    private matMul: Matmul;
    public activationFn: TensorOperation;
    constructor(num_inputs: number = 1, num_out: number = 1, public activation: String | null = null, private useBias = true) {
        super();
        this.weights = TensorUtils.rand([num_out, num_inputs], -1e-3, 1e-3);
        if (useBias) this.biases = TensorUtils.rand([1, num_out], -1e-3, 1e-3);
        this.registerParameter('weights', this.weights);
        this.registerParameter('biases', this.biases);     

        this.transposer = new Transpose();
        this.adder = new Add();
        this.matMul = new Matmul();
    }

    setupActivations(){
        if(this.activation && !this.activationFn) this.activationFn = eval(`new ${this.activation}`);
    }
    public forward(input: Tensor): Tensor {
        this.setupActivations();

        let x = this.matMul([input, this.transposer([this.weights])]);
        if (this.useBias) x = this.adder([x, this.biases]);
        
        if (this.activation)
            x = this.activationFn([x]);

        return x;
    }

}
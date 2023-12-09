/// <reference path="Layer.ts" />
// a simple impelentation of Linear Layer

class Linear extends Layer {
    private weights: Tensor;
    private biases: Tensor;
    private transposer: Transpose;
    private adder: Add;
    private matMul: Matmul;
    constructor(num_inputs: number, num_out: number, private activation: "relu" | null = null) {
        super();
        this.weights = TensorUtils.filled([num_out, num_inputs], 0);
        this.biases = TensorUtils.filled([1, num_out], 0);
        this.registerParameter('weights', this.weights);
        this.registerParameter('biases', this.biases);

        this.transposer = new Transpose();
        this.adder = new Add();
        this.matMul = new Matmul();

    }
    public forward(input: Tensor): Tensor {
        let x = this.matMul([input, this.transposer([this.weights])]);
        x = this.adder([x, this.biases]);
        if (this.activation == 'relu') {
            x = new Max()([x, new Tensor([0])])
        }
        
        return x;
    }
}
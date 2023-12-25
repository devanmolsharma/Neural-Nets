abstract class Model extends Function {
    private _bound: Model;
    constructor() {
        // Call the super constructor to set up the function
        super('...args', 'return this._bound._call(...args)');

        // Bind the instance to enable chaining
        this._bound = this.bind(this);

        // Return the bound instance
        return this._bound;
    }

    _call(...params: any[]) {
        return this.forward(params);
    }

    isPrimitive(val: Object): boolean {

        if (val === Object(val)) {
            return false;
        } else {
            return true;
        }
    }

    public toJson() {
        let j: Array<any> = [];
        (Object.getOwnPropertyNames(this)).forEach((name: string) => {
            const value: any = (this as any)[name];
            let tempValue: any;
            if (value instanceof Tensor) {
                tempValue = value.value;
            }
            else if (value instanceof Array && value[0] && value[0] instanceof Tensor) {
                tempValue = { array: value.map((e) => e.value) };
            }
            else if (value instanceof Layer) {
                tempValue = value.toJson();
            }
            else if (value instanceof Array && value[0] && value[0] instanceof Layer) {
                tempValue = { array: value.map((e) => e.toJson()) };
            }
            else if (!this.isPrimitive(value)) {
                return;
            } else {
                tempValue = value;
            }
            j.push({
                name: name,
                value: tempValue
            })
        });
        return j;
    }

    public loadData(json: any) {
        let params = json;
        params.forEach((data: any) => {
            if (data.name === 'length') return;
            if (data.name === 'name') return;
            if (data.value instanceof Array && this.isPrimitive(data.value.flat[0])) {
                (this as any)[data.name] = new Tensor(data.value);

            }
            else if (data.value.layer) {

                const obj = eval(`new ${data.value.layer}()`);
                obj.loadData(data.value)
                    (this as any)[data.name] = obj;


            }
            else if (data.value.array) {
                let temp: any[] = [];
                data.value.array.forEach((e: any) => {
                    if (!(e.value instanceof Array)) {
                        const jsn = JSON.parse(e);
                        const obj = eval(`new ${jsn.layer}()`);
                        obj.loadData(jsn)
                        temp.push(obj)
                    } else {
                        const tensor = new Tensor(e);
                        temp.push(tensor);
                    }

                });

                (this as any)[data.name] = temp;


            }

            else {
                (this as any)[data.name] = data.value;
            }
        });

        return this;
    }

    abstract forward(params: any[]): Tensor;

    abstract getParameters(): Map<String, Tensor>[];

}
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;
using NeuralNetworkExceptions;

namespace Nork {
    public abstract class NeuralNetworkSaver {
        public static async Task SaveAsync( NeuralNetwork neuralNet, string path = "saves/neuralnetwork.nns" ) {
            try {
                BinaryFormatter formatter = new BinaryFormatter();

                using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                    formatter.Serialize(fs, neuralNet);
                }
            }
            catch {}
            await Task.Delay(0);
        }
        public static void Save( NeuralNetwork neuralNet, string path = "saves/neuralnetwork.nns" ) {
            try {
                BinaryFormatter formatter = new BinaryFormatter();

                using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                    formatter.Serialize(fs, neuralNet);
                }
            }
            catch {}
        }
        public static NeuralNetwork Get( string path = "saves/neuralnetwork.nns" ) {
            try {
                BinaryFormatter formatter = new BinaryFormatter();

                using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                    return ( NeuralNetwork ) formatter.Deserialize(fs);
                }
            }
            catch {
                return null;
            }
        }
        public static async Task< NeuralNetwork > GetAsync( string path = "saves/neuralnetwork.nns" ) {
            await Task.Delay(0);
            try {
                BinaryFormatter formatter = new BinaryFormatter();

                using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                    return ( NeuralNetwork ) formatter.Deserialize(fs);
                }
            }
            catch {
                return null;
            }
        }
    }
    
    [Serializable]
    public class NeuralNetwork {
        public List<double> neuralNetworkOutput {
            get => neuralNetwork_output;
        }
        public List<Layer> MiddleLayers {
            get => middle_layers;
        }
        public Layer InputLayer {
            get => input_layer;
        }
        public Layer OutputLayer {
            get => output_layer;
        }
        public neuralNetworkOptions Options {
            get => options;
            set {
                options = value;
            }
        }
        private Layer input_layer;
        private List<Layer> middle_layers = new List<Layer>();
        private List<Layer> recursive_connector_layers = new List<Layer>();
        private Layer output_layer;
        private List<double> neuralNetwork_output;
        private neuralNetworkOptions options;

        public NeuralNetwork( neuralNetworkOptions newOptions ) {
            options = newOptions;
            switch ( options.Type ) {
                case neuralNetworkOptions.NeuralNetworkType.Perceptron:
                    GeneratePerceptron();
                    break;

                case neuralNetworkOptions.NeuralNetworkType.DeepPerceptron:
                    GenerateDeepPerceptron();
                    break;

                case neuralNetworkOptions.NeuralNetworkType.AutoEncoder:
                    GenerateAutoEncoder();
                    break;

                case neuralNetworkOptions.NeuralNetworkType.DCNN:
                    GenerateDCNN();
                    break;
            }
        }
        public NeuralNetwork() {}

        private void GenerateAutoEncoder() {
            if ( options.UseInputLayer ) {
                input_layer = new Layer( "input", options.InputNeuronsCount, options.UseBiasNeurons );
            }
            middle_layers.Add( new Layer( "middle", options.MiddleNeuronsCount, options.UseBiasNeurons ) );
            if ( options.UseOutputLayer ) {
                output_layer = new Layer( "output", options.OutputNeuronsCount, options.UseBiasNeurons );
            }
        }
        private void GenerateDCNN() {
            if ( options.UseInputLayer ) {
                input_layer = new Layer( "input", options.InputNeuronsCount, options.UseBiasNeurons );
            }
            for (int i = options.MiddleNeuronsCount; i > 1; i--) {
                middle_layers.Add( new Layer( "middle", i, options.UseBiasNeurons ) );
            }
            if ( options.UseOutputLayer ) {
                output_layer = new Layer( "output", options.OutputNeuronsCount, options.UseBiasNeurons );
            }
            SetDependencies();
        }
        private void GenerateDeepPerceptron() {
            if ( options.UseInputLayer ) {
                input_layer = new Layer( "input", options.InputNeuronsCount, options.UseBiasNeurons );
            }
            for (int i = 0; i < options.MiddleLayersCount; i++) {
                middle_layers.Add( new Layer( "middle", options.MiddleNeuronsCount, options.UseBiasNeurons ) );
            }
            if ( options.UseOutputLayer ) {
                output_layer = new Layer( "output", options.OutputNeuronsCount, options.UseBiasNeurons );
            }

            SetDependencies();
        }
        private void GeneratePerceptron() {
            if ( options.UseInputLayer ) {
                input_layer = new Layer( "input", options.InputNeuronsCount, options.UseBiasNeurons );
            }
            if ( options.UseOutputLayer ) {
                output_layer = new Layer( "output", options.OutputNeuronsCount, options.UseBiasNeurons );
            }

            SetDependencies();
        }
        private void SetDependencies() {
            if ( middle_layers.Count > 0 ) {
                if ( options.UseInputLayer ) {
                    input_layer.SetNextLayer( middle_layers[0] );
                }

                var lastMiddleLayerIndex = middle_layers.Count - 1;
                if ( options.UseOutputLayer ) {
                    middle_layers[ lastMiddleLayerIndex ].SetNextLayer( output_layer );
                }
            }
            else {
                if ( options.UseInputLayer & options.UseOutputLayer ) {
                    input_layer.SetNextLayer( output_layer );
                }
            }
            for (int i = 0; i < middle_layers.Count - 1 ; i++) {
                middle_layers[i].SetNextLayer( middle_layers[ i + 1 ] );
            }
        }
        private void SetInputValues( List< float > inputValues ) {
            if ( options.UseInputLayer ) {
                input_layer.SetInputValues( inputValues );
            }
        }
        public void SetWeights() {
            if ( options.UseInputLayer ) {
                input_layer.SetWeights();
            }
            if ( options.UseOutputLayer ) {
                output_layer.SetWeights();
            }
            for (int i = 0; i < middle_layers.Count; i++) {
                middle_layers[i].SetWeights();
            }
        }
        public void SetInputLayer( Layer layer ) {
            this.input_layer = layer;
        }
        public void SetOuputLayer( Layer layer ) {
            this.output_layer = layer;
        }
        private void ClearInputValues() {
            if ( options.UseInputLayer ) {
                input_layer.ClearInputValues();
            }
            if ( options.UseOutputLayer ) {
                output_layer.ClearInputValues();
            }
            for (int i = 0; i < middle_layers.Count; i++) {
                middle_layers[i].ClearInputValues();
            }
        }
        private void GetOutValues() {
            neuralNetwork_output = output_layer.GetOutValues();
        }
        public void Work( List< float > inputValues = null ) {
            if ( inputValues != null ) {
                SetInputValues( inputValues );
            }
            if ( options.UseInputLayer ) {
                input_layer.Work();
            }
            for (int i = 0; i < middle_layers.Count; i++) {
                middle_layers[i].Work();
            }
            if ( options.UseOutputLayer ) {
                output_layer.Work();
                GetOutValues();
            }
            ClearInputValues();
        }
        public void Learn( List< float > idealValues = null ) {
            if ( options.UseOutputLayer & idealValues != null ) {
                output_layer.Learn( idealValues, options.LearningSpeed, options.Moment );
            }
            for (int i = middle_layers.Count - 1; i > -1; i--) {
                middle_layers[i].Learn( idealValues, options.LearningSpeed, options.Moment );
            }
            if ( options.UseInputLayer ) {
                input_layer.Learn( idealValues, options.LearningSpeed, options.Moment );
            }
        }
    }
    [Serializable]
    public class Layer {
        private List<Neuron> neurons = new List<Neuron>();
        private string type;
        private bool use_bias_neurons;

        public Layer( string layerType, int neuronsCount, bool useBiasNeurons = true ) {
            type = layerType;
            use_bias_neurons = useBiasNeurons;
            for (int i = 0; i< neuronsCount; i++) {
                neurons.Add( new Neuron( type ) );
            }
            if ( useBiasNeurons & type != "output" ) {
                neurons.Add( new Neuron( "bias" ) );
            }
        }
        public void SetNextLayer( Layer layer ) {
            for (int i = 0; i < neurons.Count; i++) {
                for (int j = 0; j < layer.neurons.Count; j++) {
                    neurons[i].NextNeurons.Add( layer.neurons[j] );
                }
            }
            for (int i = 0; i < layer.neurons.Count; i++) {
                for (int j = 0; j < neurons.Count; j++) {
                    layer.neurons[i].PriviousNeurons.Add( neurons[j] );
                }
            }
        }
        public void SetWeights() {
            for (int i = 0; i < neurons.Count; i++) {
                neurons[i].SetWeights();
            }
        }
        public void SetInputValues( List< float > inputValues ) {
            if ( use_bias_neurons ) {
                if ( neurons.Count - 1 < inputValues.Count ) {
                    throw new LayerException( "Input values count mustn't be more than neurons count." );
                }
            }
            else {
                if ( neurons.Count < inputValues.Count ) {
                    throw new LayerException( "Input values count mustn't be more than neurons count." );
                }
            }
            for (int i = 0; i < inputValues.Count; i++) {
                neurons[i].InputValue = ( double ) inputValues[i];
            }
        }
        public void ClearInputValues() {
            for (int i = 0; i < neurons.Count; i++) {
                neurons[i].ClearInputValue();
            }
        }
        public void Work() {
            for (int i = 0; i < neurons.Count; i++) {
                neurons[i].Work();
            }
        }
        public void Learn( List< float > idealValues, double learningSpeed, double moment ) {
            if ( type == "output" ) {
                if ( idealValues.Count != neurons.Count ) {
                    throw new LayerException( "Ideal values count doesn't match with neurons count." );
                }
                for (int i = 0; i < idealValues.Count; i++) {
                    neurons[i].Learn( idealValues[i], learningSpeed, moment );
                }
            }
            else {
                for (int i = 0; i < neurons.Count; i++) {
                    neurons[i].Learn( 0, learningSpeed, moment );
                }
            }
        }
        public List<double> GetOutValues() {
            var outValuesList = new List<double>();
            for (int i = 0; i < neurons.Count; i++) {
                outValuesList.Add( neurons[i].OutputValue );
            }
            return outValuesList;
        }
    }
    [Serializable]
    public class Neuron {
        public double InputValue {
                get => input_Value;
                set {
                    input_Value = value;
                }
            }
        public double OutputValue {
            get => output_Value;
            set {}
        }
        public List<Neuron> PriviousNeurons {
            get => privious_neurons;
            set {
                privious_neurons = value;
            }
        }
        public List<Neuron> NextNeurons {
            get => next_neurons;
            set {
                next_neurons = value;
            }
        }

        private List<Neuron> next_neurons = new List<Neuron>();
        private List<Neuron> privious_neurons = new List<Neuron>();
        private List<double> weights = new List<double>();
        private List<double> last_weights_delta = new List<double>();
        private double input_Value = 0;
        private double output_Value = 0;
        private double delta = 0;

        private string type = "";
        private object locker = new Object();

        public Neuron(string layerType) {
            type = layerType;
        }
        public void SetWeights() {
            weights.Clear();
            for (int i = 0;i < next_neurons.Count; i++) {
                if (next_neurons[i].type != "bias") {
                    weights.Add( new Random().NextDouble() - 0.5 );
                }
            }
        }
        
        public void Work( byte b = 0 ) {
            if ( type != "bias" ) {
                output_Value = Sigmoid(input_Value);
            }
            else {
                output_Value = input_Value;
            }

            var weightsCounter = 0;
            for (int i = 0; i < next_neurons.Count; i++) {
                if (next_neurons[i].type != "bias") {
                    if (type == "bias") {
                        next_neurons[i].input_Value += weights[ weightsCounter ];
                    }
                    else {
                        next_neurons[i].input_Value += weights[ weightsCounter ] * output_Value;
                    }
                }
                weightsCounter++;
            }
        }
        public void ClearInputValue() {
            input_Value = 0;
        }
        public void Learn(double idealValue, double learningSpeed, double moment) {
            if (type == "output") {
                delta = ( idealValue - output_Value ) * SigmoidDerivate( output_Value );
            }
            else {

                delta = SumOfMultiplyingWeightsAndDeltas( weights, next_neurons ) * SigmoidDerivate( output_Value );
                UpdateThisWeights( learningSpeed, moment );
            }
        }
        private void UpdateThisWeights( double learningSpeed, double moment ) {
            var weightsCounter = 0;

            for (int i = 0; i < next_neurons.Count; i++) {
                if ( next_neurons[i].type != "bias" ) {
                    double gradient = CalculateGradient( next_neurons[i].delta );
                    double weightDelta = CalculateWeightDelta( learningSpeed, moment, gradient, weightsCounter );
                    weights[ weightsCounter ] += weightDelta;
                    SetLastWeightDelta( weightDelta, weightsCounter );
                }
                weightsCounter++;
            }
        }
        private void SetLastWeightDelta(double thisDelta, int weightsIndex) {
            lock ( last_weights_delta ) {
                if (last_weights_delta.Count < weightsIndex + 1) {
                    last_weights_delta.Add( thisDelta );
                }
                else {
                    last_weights_delta[ weightsIndex ] = thisDelta;
                }
            }
        }
        private double CalculateWeightDelta( double learnSpeed, double moment, double gradient, int weightIndex ) {
            if (last_weights_delta.Count == weights.Count) {
                return ( learnSpeed * gradient ) + ( moment * last_weights_delta[ weightIndex ] );
            }
            else {
                return ( learnSpeed * gradient );
            }
        }
        private double CalculateGradient( double nextNeuronDelta ) {
            if (type != "bias") {
                return nextNeuronDelta * output_Value;
            }
            else {
                return nextNeuronDelta * 1;
            }
        }
        private double SumOfMultiplyingWeightsAndDeltas( List<double> weightsForMultiply, List<Neuron> deltasForMultiply ) {
            double sum = 0;
            for (int i = 0; i < weightsForMultiply.Count; i++) {
                sum += weightsForMultiply[i] * deltasForMultiply[i].delta;
            }
            return sum;
        }
        private double Sigmoid(double value) {
            return 1 / ( 1 + Math.Exp(-value) );
        }
        private double SigmoidDerivate(double value) {
            if ( type == "bias" ) {
                return 1;
            }
            else {
                return ( 1 - value ) * value;
            }
        }
    }
    [Serializable]
    public class neuralNetworkOptions {
        public NeuralNetworkType Type {
            get => type;
            set {
                type = value;
            }
        }
        public int InputNeuronsCount {
            get => inputNeuronsCount;
            set {
                if ( value > 0 ) {
                    inputNeuronsCount = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Count of neurons can't be less than 1." );
                }
            }
        }
        public int MiddleNeuronsCount {
            get => middleNeuronsCount;
            set {
                if ( value > 0 ) {
                    middleNeuronsCount = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Count of neurons can't be less than 1." );
                }
            }
        }
        public int OutputNeuronsCount {
            get => outputNeuronsCount;
            set {
                if ( value > 0 ) {
                    outputNeuronsCount = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Count of neurons can't be less than 1." );
                }
            }
        }
        public int MiddleLayersCount {
            get => middleLayersCount;
            set {
                if ( value >= 0 ) {
                    middleLayersCount = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Count of middle layers can't be negative." );
                }
            }
        }
        public bool UseBiasNeurons {
            get => useBiasNeurons;
            set {
                useBiasNeurons = value;
            }
        }
        public double LearningSpeed {
            get => learningSpeed;
            set {
                if ( value > 0 & value <= 1 ) {
                    learningSpeed = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Learning speed can't be less than 0 and more than 1." );
                }
            }
        }
        public double Moment {
            get => moment;
            set {
                if ( value > 0 & value < 1 ) {
                    moment = value;
                }
                else {
                    throw new NeuralNetworkOptionsException( "Moment can't be less than 0 and more than 1." );
                }
            }
        }
        public bool UseInputLayer {
            get => useInputLayer;
            set {
                useInputLayer = value;
            }
        }
        public bool UseOutputLayer {
            get => useOutputLayer;
            set {
                useOutputLayer = value;
            }
        }
        public enum NeuralNetworkType {
            Perceptron,
            DeepPerceptron,
            AutoEncoder,
            DCNN
        }
        private NeuralNetworkType type = NeuralNetworkType.Perceptron;
        private int inputNeuronsCount = 1;
        private int middleNeuronsCount = 1;
        private int outputNeuronsCount = 1;
        private int middleLayersCount = 1;
        private bool useBiasNeurons = true;
        private bool useInputLayer = true;
        private bool useOutputLayer = true;
        private double learningSpeed = 0.7;
        private double moment = 0.3;
    }

    namespace ErrorController {
        public abstract class Error {
            public static double MSE( List< float > idealValues, List< double > actualValues ) {
                double outError = 0;
                var i = 0;

                for ( i = 0; i < actualValues.Count; i++ ) {
                    if ( idealValues.Count > i ) {
                        outError += Math.Pow( idealValues[i] - actualValues[i], 2 );
                    }
                    else {
                        outError += Math.Pow( 0 - actualValues[i], 2 );
                    }
                }
                if ( !( outError >= 0 ) ) {
                        Console.WriteLine( "error" );
                    }
                return outError / i;
            }
        }
    }
}
namespace NeuralNetworkExceptions {
    internal class LayerException : Exception {
        internal LayerException(string message)
        : base(message) { }
    }
    internal class NeuralNetworkOptionsException : Exception {
        internal NeuralNetworkOptionsException(string message)
        : base(message) { }
    }
}

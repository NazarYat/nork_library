using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;
using DatasetExceptions;

namespace Nork {
    namespace DatasetCreator {
        [Serializable]
        public class Dataset {
            public List<Frame> Frames {
                get => frames;
            }
            private List<Frame> frames = new List<Frame>();

            public void GenerateDatasetFromNumberLine( List<double> numberLine, int visibleArea, int predictionArea) {
                if ( numberLine.Count < ( visibleArea + predictionArea ) ) {
                    throw new DatasetException( "Number line count must be less than sum of visible area and prediction area." );
                }

                var cell = visibleArea + predictionArea;

                for ( int j = 0; j < numberLine.Count - cell; j++ ) {
                    
                    var realVisibleValues = new List< float >();
                    var realPredictionValues = new List< float >();
                    var predictionValues = new List< float >();

                    var localizedDataList = new List< float >();
                    var frameDataList = new List< double >();

                    for ( int i = 0; i < cell; i++ ) {
                        frameDataList.Add( numberLine[ i + j ] );
                    }

                    localizedDataList = GetLocalizegDoubleStringInFloat( frameDataList, visibleArea );

                    for ( int i = 0; i < visibleArea; i++ ) {
                        realVisibleValues.Add( localizedDataList[ i ] );
                    }
                    for ( int i = visibleArea; i < cell; i++ ) {
                        realPredictionValues.Add( ( float ) numberLine[ i + j ] );
                    }

                    var lastVisibleValue = realVisibleValues[ realVisibleValues.Count - 1 ];
                    var minPredictionValue = GetMinFloatValue( realPredictionValues );
                    var maxPredictionValue = GetMaxFloatValue( realPredictionValues );

                    if ( ( lastVisibleValue - minPredictionValue ) / ( lastVisibleValue / 100 ) > 0.4 ) {
                        predictionValues.Add( 1 );
                    }
                    else {
                        predictionValues.Add( 0 );
                    }
                    if ( ( lastVisibleValue - maxPredictionValue ) / ( lastVisibleValue / 100 ) < -0.4 ) {
                        predictionValues.Add( 1 );
                    }
                    else {
                        predictionValues.Add( 0 );
                    }
                    if ( predictionValues[0] == 0 & predictionValues[1] == 0 ) {
                        predictionValues.Add( 1 );
                    }
                    else {
                        predictionValues.Add( 0 );
                    }

                    frames.Add( new Frame( realVisibleValues, predictionValues ) );
                
                    for ( int i = 0; i < realVisibleValues.Count; i++ ) {
                        if ( !( realVisibleValues[ i ] > -30 ) ) {
                            Console.WriteLine( realVisibleValues[i] );
                        }
                    }
                    for ( int i = 0; i < predictionValues.Count; i++ ) {
                        if ( realVisibleValues[i] == float.NaN ) {
                            Console.WriteLine( "" );
                        }
                    }
                }
            }
            float GetMaxFloatValue( List< float > sourceList ) {
                var maxValue = 0.0f;
                for ( int i = 0; i < sourceList.Count; i++ ) {
                    if ( sourceList[i] > maxValue ) {
                        maxValue = sourceList[i];
                    }
                }
                return maxValue;
            }
            float GetMinFloatValue( List< float > sourceList ) {
                if ( sourceList.Count > 0 ) {
                    var minValue = sourceList[0];
                    for ( int i = 0; i < sourceList.Count; i++ ) {
                        if ( sourceList[i] < minValue ) {
                            minValue = sourceList[i];
                        }
                    }
                    return minValue;
                }
                return 0;
            }
            List< float > GetLocalizegDoubleStringInFloat( List< double > sourceList, int visibleArea ) {
            
                var maxValue = 0.0;
                var outList = new List< float >();

                for ( int i = 0; i < visibleArea; i++ ) {
                    if ( sourceList[i] > maxValue ) {
                        maxValue = sourceList[i];
                    }
                }
                var minValue = maxValue;
                for ( int i = 0; i < visibleArea; i++ ) {
                    if ( sourceList[i] < minValue ) {
                        minValue = sourceList[i];
                    }
                }
                var valuePerPercent = ( maxValue - minValue );
                for ( int i = 0; i < sourceList.Count; i++ ) {
                    var delta = sourceList[i] - minValue;
                    var value = 0.0;
                    if ( valuePerPercent != 0 ) {
                        value = delta / valuePerPercent;
                    }
                    outList.Add( (float) ( value - 0.5 ) );
                }

                return outList;
            }
            string DecimalToBinary(int decimalNumber) {
                var binaryNumber = string.Empty;
                while (decimalNumber > 0)
                {
                    binaryNumber = (decimalNumber % 2) + binaryNumber;
                    decimalNumber /= 2;
                }

                return binaryNumber;
            }
            public void Shuffle() {
                var rand = new Random();
                for ( int i = 0; i < this.frames.Count; i++ ) {
                    var j = rand.Next( 0, this.frames.Count - 1 );
                    var timedValue = this.frames[j];
                    this.frames[j] = this.frames[i];
                    this.frames[i] = timedValue;
                }
            }
            [Serializable]
            public class Frame {
                public List< float > InputValues {
                    get => input_values;
                }
                public List< float > IdealValues {
                    get => ideal_values;
                }
                private List< float > input_values = new List< float >();
                private List< float > ideal_values = new List< float >();

                public Frame( List< float > inputValues, List< float > outputValues ) {
                    for (int i = 0; i < inputValues.Count; i++) {
                        input_values.Add( inputValues[i] );
                    }
                    for (int i = 0; i < outputValues.Count; i++) {
                        ideal_values.Add( outputValues[i] );
                    }
                }
            }
        }
        public abstract class DatasetSaver {
            static void CreateSavesFolderIfNotExista() {
                if ( !Directory.Exists("saves") ) {
                    Directory.CreateDirectory("saves");
                }
            }
            public static async Task SaveAsync( Dataset dataSet, string path = "saves/dataset.dts" ) {
                try {
                    CreateSavesFolderIfNotExista();
                    BinaryFormatter formatter = new BinaryFormatter();

                    using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                        formatter.Serialize(fs, dataSet);
                    }
                }
                catch {}
                await Task.Delay(0);
            }
            public static void Save( Dataset dataSet, string path = "saves/dataset.dts" ) {
                try {
                    CreateSavesFolderIfNotExista();
                    BinaryFormatter formatter = new BinaryFormatter();

                    using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                        formatter.Serialize(fs, dataSet);
                    }
                }
                catch {}
            }
            public static Dataset Get( string path = "saves/dataset.dts" ) {
                try {
                    CreateSavesFolderIfNotExista();
                    BinaryFormatter formatter = new BinaryFormatter();

                    using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                        return ( Dataset ) formatter.Deserialize(fs);
                    }
                }
                catch {
                    return new Dataset();
                }
            }
            public static async Task< Dataset > GetAsync( string path = "saves/dataset.dts" ) {
                await Task.Delay(0);
                try {
                    CreateSavesFolderIfNotExista();
                    BinaryFormatter formatter = new BinaryFormatter();

                    using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate)) {
                        return ( Dataset ) formatter.Deserialize(fs);
                    }
                }
                catch {
                    return new Dataset();
                }
            }
        }
    }
}
namespace DatasetExceptions {
    internal class DatasetException : Exception {
        internal DatasetException(string message)
        : base(message) { }
    }
}
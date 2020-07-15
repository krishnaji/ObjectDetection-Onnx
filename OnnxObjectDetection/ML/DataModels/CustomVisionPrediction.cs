using Microsoft.ML.Data;

namespace OnnxObjectDetection
{
    public class CustomVisionPrediction : IOnnxObjectPrediction
    {
        [ColumnName("classLabel")]
        public string[] ClassLabel { get; set; }

        //[ColumnName("loss")]
        //public float[] Loss { get; set; }

        public float[] PredictedLabels { get; set; }
    }
}

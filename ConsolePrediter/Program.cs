using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using OnnxObjectDetection;

namespace ConsolePrediter
{
    class Program
    {
        private static readonly string modelsDirectory = Path.Combine(Environment.CurrentDirectory, @"ML\OnnxModels");

        static void Main(string[] args)
        {
            Program p = new Program();

            p.Run();
        }

        public void Run()
        {
            this.LoadModel();

            string filePath = @"C:\Users\shkhose\Downloads\ImageDetectionForKrishna\ImageDetectionForKrishna\IMG_1538.JPG";
            ImageInputData inputData = this.LoadImageFile(filePath);

            CustomVisionPrediction prediction = this._customVisionPredictionEngine.Predict(inputData);
            Console.WriteLine(prediction.ClassLabel[0]);

        }

        private PredictionEngine<ImageInputData, CustomVisionPrediction> _customVisionPredictionEngine;
        private OnnxOutputParser _outputParser;

        private void LoadModel()
        {
            // Check for an Onnx model exported from Custom Vision
            var customVisionExport = Directory.GetFiles(modelsDirectory, "*.zip").FirstOrDefault();


            var customVisionModel = new CustomVisionModel(customVisionExport);
            var modelConfigurator = new OnnxModelConfigurator(customVisionModel);

            this._outputParser = new OnnxOutputParser(customVisionModel);
            this._customVisionPredictionEngine = modelConfigurator.GetMlNetPredictionEngine<CustomVisionPrediction>();
        }

        private ImageInputData LoadImageFile(string imageFilePath)
        {
            Image image = Image.FromFile(imageFilePath);
            //Convert to Bitmap
            var bitmapImage = (Bitmap)image;

            //Set the specific image data into the ImageInputData type used in the DataView
            ImageInputData imageInputData = new ImageInputData { Image = bitmapImage };
            return imageInputData;
        }

    }
}

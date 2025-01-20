//
//  ContentView.swift
//  AI
//
//  Created by Autotoll Developer on 20/1/2025.
//

import SwiftUI
import CoreML
import Vision



struct ContentView: View {
    @State private var image: UIImage?
    @State private var prediction: String = "請選擇一張圖片進行推理"
    @State private var isLoading: Bool = false // Loading 狀態

    var body: some View {
        VStack {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
            } else {
                Text("請選擇一張圖片")
                    .padding()
            }
            
            if isLoading {
                ProgressView("正在處理...")
                    .padding()
            }
            
            Button(action: {
                selectImage()
            }) {
                Text("選擇圖片")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            
            Text(prediction)
                .padding()
            
            Spacer()
        }
        .sheet(isPresented: $showingImagePicker, onDismiss: loadImage) {
            ImagePicker(image: $inputImage)
        }
    }
    
    @State private var inputImage: UIImage?
    @State private var showingImagePicker = false
    
    func selectImage() {
        showingImagePicker = true
        print("用戶選擇圖片")
    }
    
    func loadImage() {
        guard let inputImage = inputImage else {
            print("未選擇圖片")
            return
        }
        image = inputImage
        print("圖片已加載，開始模型推理")
        runModel(on: inputImage)
    }
    
    func runModel(on image: UIImage) {
        isLoading = true
        print("加載模型...")
        
        guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
            prediction = "無法加載模型"
            isLoading = false
            print("模型加載失敗")
            return
        }
        
        let request = VNCoreMLRequest(model: model) { (request, error) in
            if let error = error {
                print("推理過程中出現錯誤: \(error.localizedDescription)")
                self.isLoading = false
                return
            }

            if let results = request.results as? [VNClassificationObservation], let topResult = results.first {
                DispatchQueue.main.async {
                    self.prediction = "分類: \(topResult.identifier), 信心: \(topResult.confidence)"
                    self.isLoading = false
                }
            } else {
                print("無法獲取結果")
                self.isLoading = false
            }
        }
        
        guard let resizedImage = image.resize(to: CGSize(width: 640, height: 640)) else {
            print("無法調整圖片大小")
            return
        }

        
        guard let ciImage = CIImage(image: resizedImage) else {
            prediction = "無法處理圖片"
            isLoading = false
            print("圖片轉換失敗")
            return
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                print("開始執行模型推理...")
                try handler.perform([request])
            } catch {
                print("執行模型推理失敗: \(error)")
                DispatchQueue.main.async {
                    self.isLoading = false
                }
            }
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker
        
        init(parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
    
    @Environment(\.presentationMode) var presentationMode
    @Binding var image: UIImage?
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}

#Preview {
    ContentView()
}


extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
}

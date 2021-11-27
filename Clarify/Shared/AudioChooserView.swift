//
//  AudioChooser.swift
//  Clarify (iOS)
//
//  Created by ojaswee c on 11/27/21.
//

import SwiftUI

struct AudioChooserView: View {
    
    @ObservedObject var audioRecorder: AudioRecorder
        
    //list of previous recordings
    //start and stop button
    var body: some View {
        
        VStack {
                    RecordingsList(audioRecorder: audioRecorder)
                    //...
                }
        
        
        VStack {
            if audioRecorder.recording == false {
                Button(action: {print("Start recording")}) {
                    Image(systemName: "circle.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 100, height: 100)
                        .clipped()
                        .foregroundColor(.red)
                        .padding(.bottom, 40)
                }
            } else {
                Button(action: {print("Stop recording)")}) {
                    Image(systemName: "stop.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 100, height: 100)
                        .clipped()
                        .foregroundColor(.red)
                        .padding(.bottom, 40)
                }
            }
        }
    }
}

struct AudioChooserView_Previews: PreviewProvider {
    static var previews: some View {
        AudioChooserView(audioRecorder: AudioRecorder())
    }
}

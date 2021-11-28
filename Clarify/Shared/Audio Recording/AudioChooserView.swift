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
                Button(action: {self.audioRecorder.startRecording()}) {
                    Image(systemName: "circle.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: 60)
                        .clipped()
                        .foregroundColor(.red)
                        .padding(.bottom, 40)
                        .padding(.top, 10)
                        .overlay(
                                Circle()
                                    .stroke(Color.black, lineWidth: 3)
                                    .padding(.bottom, 40)
                                    .padding(.top, 10)
                            )
                }
            } else {
                Button(action: {self.audioRecorder.stopRecording()}) {
                    Image(systemName: "stop.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 60, height: 60)
                        .clipped()
                        .foregroundColor(.red)
                        .padding(.bottom, 40)
                        .padding(.top, 10)
                        .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.black, lineWidth: 3)
                                    .aspectRatio(1.0, contentMode: .fit)
                                    .padding(.bottom, 40)
                                    .padding(.top, 10)
                            )
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

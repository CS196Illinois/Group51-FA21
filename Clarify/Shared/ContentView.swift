//
//  ContentView.swift
//  Shared
//
//  Created by ojaswee c on 11/26/21.
//

import SwiftUI


let contentView = ContentView(audioRecorder: AudioRecorder())

struct ContentView: View {
    
    @ObservedObject var audioRecorder: AudioRecorder
    
    var body: some View {
        
        //logo display, first screen
        Text("**Clarify**")
            .padding()
        
        //list of previous recordings
        VStack {
                    RecordingsList(audioRecorder: audioRecorder)
                    //...
                }
        
        //start and stop button
        VStack {
                    AudioChooserView(audioRecorder: audioRecorder)
                    //...
                }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(audioRecorder: AudioRecorder())
    }
}

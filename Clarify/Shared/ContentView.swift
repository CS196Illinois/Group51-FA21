//
//  ContentView.swift
//  Shared
//
//  Created by ojaswee c on 11/26/21.
//

import SwiftUI


//need to separate launch screen from audio chooser
//https://www.youtube.com/watch?v=NLIx0q3OixQ for audio choosing
//core data

struct ContentView: View {
    
    @ObservedObject var audioRecorder: AudioRecorder
    
    var body: some View {
        
        NavigationView {
            
            VStack {
                //Text("**Press to record**").padding()
                AudioChooserView(audioRecorder: audioRecorder)
            }
            
            .navigationBarTitle("1. Record your audio ")
            .navigationBarItems(trailing: EditButton())
        }
        
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(audioRecorder: AudioRecorder())
    }
}


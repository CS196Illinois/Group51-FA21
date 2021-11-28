//
//  RecordingList.swift
//  Clarify (iOS)
//
//  Created by ojaswee c on 11/27/21.
//

import SwiftUI

struct RecordingsList: View {
    
    @ObservedObject var audioRecorder: AudioRecorder
    
    var body: some View {
        List {
            ForEach(audioRecorder.recordings, id: \.createdAt) { recording in
            RecordingRow(audioURL: recording.fileURL)
            }
        }
    }
}

struct RecordingRow: View {
    
    var audioURL: URL
        
    var body: some View {
        HStack {
            //assign each audio the path of its audio file
            Text("\(audioURL.lastPathComponent)")
            //push to left
            Spacer()
        }
    }
}

struct RecordingsList_Previews: PreviewProvider {
    static var previews: some View {
        RecordingsList(audioRecorder: AudioRecorder())
    }
}

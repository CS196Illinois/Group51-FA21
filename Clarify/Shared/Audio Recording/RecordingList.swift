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
            .onDelete(perform: delete)
        }
    }
    
    //allows the list to be edited
    func delete(at offsets: IndexSet) {
            
            var urlsToDelete = [URL]()
            for index in offsets {
                urlsToDelete.append(audioRecorder.recordings[index].fileURL)
            }
        audioRecorder.deleteRecording(urlsToDelete: urlsToDelete)
        }
}

struct RecordingRow: View {
    
    var audioURL: URL
    
    @ObservedObject var audioPlayer = AudioPlayer()
        
    var body: some View {
        HStack {
            //assign each audio the path of its audio file
            Text("\(audioURL.lastPathComponent)")
            //push to left
            Spacer()
            if audioPlayer.isPlaying == false {
                Button(action: {
                    print("Start playing audio")
                }) {
                    Image(systemName: "play.circle")
                        .imageScale(.large)
                }
            } else {
                Button(action: {
                    print("Stop playing audio")
                }) {
                    Image(systemName: "stop.fill")
                        .imageScale(.large)
                }
            }
        }
    }
}

struct RecordingsList_Previews: PreviewProvider {
    static var previews: some View {
        RecordingsList(audioRecorder: AudioRecorder())
    }
}

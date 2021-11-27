//
//  AudioRecorder.swift
//  Clarify
//
//  Created by ojaswee c on 11/26/21.
//

import Foundation
import SwiftUI
import Combine
import AVFoundation

//class that records new files and saves them
class AudioRecorder: ObservableObject {
    
    //to notify observing views about changes
    let objectWillChange = PassthroughSubject<AudioRecorder, Never>()
    
    var audioRecorder: AVAudioRecorder!
    
    //objectWillChange is a suitable variable that will change when recording is changed (i.e. finished)
    var recording = false {
            didSet {
                objectWillChange.send(self)
            }
        }
}

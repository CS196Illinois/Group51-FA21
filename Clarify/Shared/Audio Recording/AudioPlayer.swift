//
//  AudioPlayer.swift
//  Clarify
//
//  Created by ojaswee c on 11/27/21.
//

import Foundation
import SwiftUI
import Combine
import AVFoundation

class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    
    //this notifies observing views about changes  (if an audio is being played)
    let objectWillChange = PassthroughSubject<AudioPlayer, Never>()

    var isPlaying = false {
        didSet {
            objectWillChange.send(self)
        }
    }
    
    var audioPlayer: AVAudioPlayer!
    
    func startPlayback (audio: URL) {
    
        let playbackSession = AVAudioSession.sharedInstance()
        do {
                try playbackSession.overrideOutputAudioPort(AVAudioSession.PortOverride.speaker)
            } catch {
                print("Playing over the device's speakers failed")
            }
        do {
                audioPlayer = try AVAudioPlayer(contentsOf: audio)
                audioPlayer.delegate = self
                audioPlayer.play()
                isPlaying = true
            } catch {
                print("Playback failed.")
            }
        
    }
    
    func stopPlayback() {
        audioPlayer.stop()
        isPlaying = false
    }
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
            if flag {
                isPlaying = false
            }
        }
    
    /*start and stop playback doesnt actually work
     if audioPlayer.isPlaying == false {
                     Button(action: {
                         self.audioPlayer.startPlayback(audio: self.audioURL)
                     }) {
                         Image(systemName: "play.circle")
                             .imageScale(.large)
                     }
                 } else {
                     Button(action: {
                         self.audioPlayer.stopPlayback()
                     }) {
                         Image(systemName: "stop.fill")
                             .imageScale(.large)
                     }
                 }
     */
    
}

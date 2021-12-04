//
//  Helper.swift
//  Clarify (iOS)
//
//  Created by ojaswee c on 11/27/21.
//

import Foundation

//gets the date based on when the file was created
func getCreationDate(for file: URL) -> Date {
    if let attributes = try? FileManager.default.attributesOfItem(atPath: file.path) as [FileAttributeKey: Any],
        let creationDate = attributes[FileAttributeKey.creationDate] as? Date {
        return creationDate
    } else {
        //returns current date if ^^ doesnt work
        return Date()
    }
}

//
//  Extension.swift
//  Clarify (iOS)
//
//  Created by ojaswee c on 11/27/21.
//

import Foundation

extension Date
{
    func toString( dateFormat format  : String ) -> String
    {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = format
        return dateFormatter.string(from: self)
    }

}

//
//  SwiftUIView.swift
//  Clarify (iOS)
//
//  Created by Utkarsh Prasad on 12/3/21.
//

import SwiftUI

struct EditableListView: View {

    @State var items: [String] = []
    @State var selections: Set<String> = []

//    @Environment(\.editMode) private var editMode: Binding<EditMode>

    var body: some View {
        List(items, id: \.self, selection: $selections) { item in
            Text(item)
        }
        .navigationBarItems(trailing:
            Button(action: {
//                self.editMode?.value.toggle()
            }) {
//                Text(self.editMode?.value == .active ? "Done" : "Edit")
            }
        )
        .animation(.default)
    }
}

extension EditMode {

    mutating func toggle() {
        self = self == .active ? .inactive : .active
    }
}


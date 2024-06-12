Sub ScheduleMeeting()
    Dim olApp As Object
    Dim olNs As Object
    Dim olAppt As Object

    ' Outlookアプリケーションを取得
    On Error Resume Next
    Set olApp = GetObject(, "Outlook.Application")
    If olApp Is Nothing Then
        Set olApp = CreateObject("Outlook.Application")
    End If
    On Error GoTo 0

    ' Outlookネームスペースを取得
    Set olNs = olApp.GetNamespace("MAPI")

    ' 新しい予定を作成
    Set olAppt = olApp.CreateItem(1) ' 1はolAppointmentItemを示す

    ' Excelのセルから情報を取得
    Dim subject As String
    Dim location As String
    Dim startTime As Date
    Dim endTime As Date
    Dim body As String

    ' セルの値を読み取る（例：A1からE1）
    subject = ThisWorkbook.Sheets("Sheet1").Range("A1").Value
    location = ThisWorkbook.Sheets("Sheet1").Range("B1").Value
    startTime = ThisWorkbook.Sheets("Sheet1").Range("C1").Value
    endTime = ThisWorkbook.Sheets("Sheet1").Range("D1").Value
    body = ThisWorkbook.Sheets("Sheet1").Range("E1").Value

    ' 予定のプロパティを設定
    With olAppt
        .Subject = subject
        .Location = location
        .Start = startTime
        .End = endTime
        .Body = body
        .ReminderSet = True
        .ReminderMinutesBeforeStart = 15
        .BusyStatus = 2 ' 2はBusyを示す
        .Save
    End With

    ' メッセージボックスで確認
    MsgBox "ミーティングが作成されました: " & subject

    ' オブジェクトをクリーンアップ
    Set olAppt = Nothing
    Set olNs = Nothing
    Set olApp = Nothing
End Sub

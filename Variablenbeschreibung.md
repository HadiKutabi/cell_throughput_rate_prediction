Variablenbeschreibung

- timestamp_ms ist ein Unix Timestamp, welcher die Umrechnung in eine konkrete Uhrzeit ermöglicht (siehe https://www.unixtimestamp.com/)

- altitude_m die Höhe über Normalnull

- veclocity ist in meter/s und acceleration in meter/s^2: Für die Beschleunigung sind negative Werte durch Bremsen möglich (Änderung der Geschwindigkeit)

- direction ist die Richtung in Grad, Norden entspricht 0 - isRegistered gibt an, ob das Gerät eine aktive Verbindung zu einer LTE Zelle hat

- rsrp ist ein Indikator für die Empfangsleistung und durch den Pfadverlust immer negativ – das heißt dann einfach, dass nur sehr geringe Leistungsmengen beim Endgerät ankommen - rsrq und rssinr sind Verhältnisse von Leistungen, auch hier sind negative werte möglich (siehe auch https://www.cablefree.net/wirelesstechnology/4glte/rsrp-rsrq-measurement-lte/)

- ss entspricht der Arbitrary Strength Unit (ASU) und ist redundant zum RSRP, da RSRP = ASU – 140

- pci ist die Physical Cell Id, welche intern vom Endgerät verwendet wird, um Codierungsaufgaben zu machen -> Sollte für euch nicht wichtig sein

- payload entspricht der übertragenden Datenmenge in Megabyte

- througput_mbits entspricht der Datenrate und somit der Zielgröße

- rtt_ms ist die Round Trip Time: Also die Signallaufzeit vom Sender zum Empfänger und wieder zurück

- txPower_dBm entspricht der Sendeleistung des Endgerätes (somit auch nur im Uplink verfügbar, weil das Endgerät im Downlink nur empfängt)

- f_mhz entspricht der Trägerfrequenz der Basisstation, daher ist dies auch in wichtiges Feature, da die Frequenz einen großen Einfluss auf die Funkausbreitungseigenschaften hat 
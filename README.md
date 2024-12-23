# FRAUD Detection

Künstliche Daten generieren

- [ ] einige wenige Beispiele aus dem Datenset als Hilfe für GPT mitgeben
- [ ] Generierung der synthetischen Daten mit mehreren Loads

Training nur auf den synthetischen Daten (positive Fälle)

- [ ] keine Daten verwenden, welche für die Generierung der synthetischen Daten verwendet werden
- [ ] Random Sample von Negativen Fraud Fällen wählen, damit das Datenset balanced ist

Klassifizierung

- [ ] mit CatBoost
- [ ] mit DL-Modell
- [ ] Initial Bias beim Modell-Training

Evtl. Vergleich CatBoostEplain mit o1-Modell-Vergleich
evtl. o1 Fragen:

- [ ] Welche Parameter sind die Hauptgründe für FRAUD
- [ ] Begründe die Fraud Fälle

Berechnung:

- [ ] Precision
- [ ] Recall
- [ ] Accuracy
- [ ] F1-Score
- [ ] (R-Squared)
-> Cofusion Matrix generieren

Vergleich von Modell unbalancedTrainingData mit Modell balancedTrainingData

## Weiteres Vorgehen

- [x] Setup GitHub Repo
- [ ] Lokales Setup Git + pytorch code ausführen.
  - *Optional: poetry verwenden*
- [ ] Alle versuchen sich an die Testdaten Generierung
  - Koordinieren welche Frauds genutzt werden.

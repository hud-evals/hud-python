/* Endpoint signature used as the title of each accordion in the REST API
   reference: a colored method label in its own aligned column, the path with
   its `{placeholders}` dimmed, then a one-line note on the same baseline.
   Colors and spacing live in custom.css. */
export const Endpoint = ({ method, path, note }) => (
  <span className="api-sig">
    <span className="api-method-col">
      <span className={`api-method api-${method.toLowerCase()}`}>{method}</span>
    </span>
    <span className="api-path">
      {path.split(/(\{[^}]+\})/).map((part, i) =>
        part.startsWith("{") ? (
          <span className="api-param" key={i}>
            {part}
          </span>
        ) : (
          part
        )
      )}
    </span>
    {note && <span className="api-note">{note}</span>}
  </span>
);
